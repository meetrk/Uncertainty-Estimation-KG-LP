from unicodedata import name
from xml.parsers.expat import model
import torch
from pathlib import Path


from model.ensemble.deep_ensemble import DeepEnsemble
from model.trainer.basepipeline import BasePipeline
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils import negative_sampling, dropout_edges
from model.calibrator.tempscaling import TemperatureEnsemble
from model.calibrator.plattscaling import PlattScalingEnsemble, PlattScalingMCDropout
from model.calibrator.isotonic import IsotonicCalibrator
from utils.evaluation import compute_uncertainty,compute_mrr_ensemble


class EnsemblePipeline(BasePipeline):
    
    def __init__(self, ensemble_model, data, config, logger):

        super().__init__(ensemble_model, data ,config, logger)

        self.ensemble = ensemble_model
        self.optimizers = self.ensemble.get_optimizers(
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.training_history.update({'ensemble_diversity': []})
    
    def start_pipeline(self):
        """Start the ensemble training pipeline."""
        max_epochs = self.train_config['epochs']
        eval_frequency = self.train_config.get('evaluation_frequency', 10)
        early_stopping = self.train_config['early stopping']['enabled']
        patience = self.train_config['early stopping'].get('patience', 10)
        delta = self.train_config['early stopping'].get('delta', 0.0)
        
        self.logger.info(f"Starting ensemble training with {self.ensemble.num_models} models for {max_epochs} epochs")
        
        best_val_mrr = -float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(1, max_epochs + 1), desc="Training Ensemble"):
            # Train all ensemble members
            ensemble_losses = self.train_ensemble()
            avg_loss = sum(ensemble_losses) / len(ensemble_losses)
            
            # Log individual model losses
            for i, loss in enumerate(ensemble_losses):
                self.writer.add_scalar(f'Loss/Model_{i}', loss, epoch)
            
            self.writer.add_scalar('Loss/Ensemble_Mean', avg_loss, epoch)
            
            self.logger.info(f'Epoch: {epoch:05d}, Avg Ensemble Loss: {avg_loss:.4f}')
            self.training_history['train_loss'].append({"epoch": epoch, "epoch_loss": avg_loss})
            
            # Evaluation
            if epoch % eval_frequency == 0:
                valid_scores, test_scores = self.test_ensemble(test=self.train_config['test'])
                
                # Compute ensemble diversity
                diversity = self.compute_diversity()
                self.training_history['ensemble_diversity'].append({
                    "epoch": epoch,
                    "diversity": diversity
                })
                self.writer.add_scalar('Ensemble/Diversity', diversity, epoch)
                
                current_val_mrr = valid_scores['mrr']
                
                # Log metrics
                self.logger.info(f"Epoch {epoch}: Ensemble Diversity = {diversity:.4f}")
                self.logger.info(f"Epoch {epoch}: Val MRR = {valid_scores['mrr']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Mean Rank = {valid_scores['mean_rank']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@1 = {valid_scores['hits@1']:.4f} ")
                self.logger.info(f"Epoch {epoch}: Val Hits@3 = {valid_scores['hits@3']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@10 = {valid_scores['hits@10']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val ECE= {valid_scores['ece']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val ACE= {valid_scores['ace']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Brier Score = {valid_scores['brier_score']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val True Probability = {valid_scores['prob_true']}")
                self.logger.info(f"Epoch {epoch}: Val Predicted Probability = {valid_scores['prob_pred']}")

                if test_scores is not None:
                    self.logger.info(f"Epoch {epoch}: Test MRR = {test_scores['mrr']:.4f}")
                    self.logger.info(f"Epoch {epoch}: Test Mean Rank = {test_scores['mean_rank']:.4f}")
                    self.logger.info(f"Epoch {epoch}: Test Hits@1 = {test_scores['hits@1']:.4f} ")
                    self.logger.info(f"Epoch {epoch}: Test Hits@3 = {test_scores['hits@3']:.4f}")
                    self.logger.info(f"Epoch {epoch}: Test Hits@10 = {test_scores['hits@10']:.4f}")
                    self.logger.info(f"Epoch {epoch}: Test ECE= {test_scores['ece']:.4f}")
                    self.logger.info(f"Epoch {epoch}: Test ACE= {test_scores['ace']:.4f}")
                    self.logger.info(f"Epoch {epoch}: Test Brier Score = {test_scores['brier_score']:.4f}")
                    self.logger.info(f"Epoch {epoch}: Test True Probability = {test_scores['prob_true']}")
                    self.logger.info(f"Epoch {epoch}: Test Predicted Probability = {test_scores['prob_pred']}")
                
                self.writer.add_scalar('MRR/Validation', valid_scores['mrr'], epoch)
                self.writer.add_scalar('Mean_Rank/Validation', valid_scores['mean_rank'], epoch)
                self.writer.add_scalar('Hits@1/Validation', valid_scores['hits@1'], epoch)
                self.writer.add_scalar('Hits@3/Validation', valid_scores['hits@3'], epoch)
                self.writer.add_scalar('Hits@10/Validation', valid_scores['hits@10'], epoch)


                self.writer.add_scalar('MC_Uncertainty/Brier_Score', valid_scores['brier_score'], epoch)
                self.writer.add_scalar('MC_Uncertainty/ECE', valid_scores['ece'], epoch)

                
                # Check for improvement
                if current_val_mrr - best_val_mrr > delta:
                    best_val_mrr = current_val_mrr
                    patience_counter = 0
                    self.logger.info(f"New best ensemble! MRR: {best_val_mrr:.4f}")
                    
                    if self.train_config['save_model']:
                        self.save_checkpoint(epoch, name=f"{self.config.get_section('dataset')['name']}_ensemble_checkpoint_{self.config.get_section('calibration')['method']}_epoch_{epoch}.pth") 
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                
                # Early stopping
                if early_stopping and patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break
                
                self.training_history['eval_metrics'].append({
                    "epoch": epoch,
                    "val_mrr": valid_scores["mrr"],
                    "val_mean_rank": valid_scores["mean_rank"],
                    "val_hits@1": valid_scores["hits@1"],
                    "val_hits@3": valid_scores["hits@3"],
                    "val_hits@10": valid_scores["hits@10"],
                    "val_brier_score": valid_scores['brier_score'],
                    "val_ece": valid_scores['ece']

                })
        
        self.writer.close()
        self.logger.info("Ensemble training completed!")
        return self.training_history
    
    def train_ensemble(self):
        """Train all ensemble members for one epoch."""
        ensemble_losses = []
        
        for model_idx in range(self.ensemble.num_models):
            model = self.ensemble.get_model(model_idx)
            optimizer = self.optimizers[model_idx]
            
            model.train()
            optimizer.zero_grad()
            
            # Edge dropout if enabled
            if self.train_config['sampling']['edge_dropout'] > 0:
                edge_index, edge_type = dropout_edges(
                    self.data.edge_index,
                    self.data.edge_type,
                    self.train_config['sampling']['edge_dropout']
                )
            else:
                edge_index, edge_type = self.data.edge_index, self.data.edge_type
            
            # Forward pass
            z = model.encode(edge_index, edge_type)
            pos_out = model.decode(z, self.data.train_edge_index, self.data.train_edge_type)
            
            # Negative sampling
            neg_edge_index, neg_edge_type = negative_sampling(self.data.train_edge_index,self.data.train_edge_type, self.data.num_nodes,1)
            neg_out = model.decode(z, neg_edge_index, neg_edge_type)
            
            # Compute loss
            out = torch.cat([pos_out, neg_out])
            gt = torch.cat([torch.full_like(pos_out, self.train_config['label_smoothing']['positive']), torch.full_like(neg_out, self.train_config['label_smoothing']['negative'])])

            
            cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
            reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
            loss = cross_entropy_loss + self.model_config['decoder']['l2_penalty'] * reg_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            
            ensemble_losses.append(float(loss))
        
        return ensemble_losses
    
    @torch.no_grad()
    def test_ensemble(self,test = True):
        """Test ensemble with uncertainty quantification."""

        self.ensemble.eval()
        
        valid_scores = compute_mrr_ensemble(
            self.data.edge_index,
            self.data.edge_type,
            self.data.valid_edge_index,
            self.data.valid_edge_type,
            self.data,
            self.ensemble
        )
        if test:
            test_scores = compute_mrr_ensemble(
                self.data.edge_index,
                self.data.edge_type,
                self.data.test_edge_index,
                self.data.test_edge_type,
                self.data,
                self.ensemble
            )
            return valid_scores, test_scores
         
        
        return valid_scores, None
    



    def load_pipeline(self, checkpoint_path, type,save):
        """Load ensemble pipeline from checkpoint."""
        self.load_checkpoint(checkpoint_path, load_optimizer=False)
        self.logger.info("Checkpoint loaded. Evaluating ensemble uncertainty on test set...")

        assert type == 'ensemble', "Loaded model is not an ensemble."

        scores = self.test_link_pred(
            type=type,
            model=self.ensemble,
            valid_edge_index=self.data.test_edge_index,
            valid_edge_type=self.data.test_edge_type)
        uncertainty_scores = self.test_uncertainty(
                model=self.ensemble,
                test_edge_index=self.data.test_edge_index,
                test_edge_type=self.data.test_edge_type,
                params={
                    'calibration_model': None,
                })
        
        if save:
            calibration_model = self.calibrate_pipeline(
                method=self.config.get_section('calibration')['method'],
                model=self.ensemble,
                max_iters=self.config.get_section('calibration').get('max_iters', 50),
                lr=self.config.get_section('calibration').get('learning_rate', 0.01)
            )

            scores = self.test_link_pred(
                type=type,
                model=self.ensemble,
                valid_edge_index=self.data.test_edge_index,
                valid_edge_type=self.data.test_edge_type,
                calibration_model=calibration_model)

            uncertainty_scores = self.test_uncertainty(
                model=self.ensemble,
                test_edge_index=self.data.test_edge_index,
                test_edge_type=self.data.test_edge_type,
                params={
                    'calibration_model': calibration_model,
                })
        
        if save:
            self.save_checkpoint(epoch=0, name=f'calibrated_{Path(checkpoint_path).name}')

        return scores

    @torch.no_grad()
    def compute_diversity(self):
        """
        Compute ensemble diversity metric.
        
        Measures how different the predictions are across ensemble members.
        Higher diversity generally leads to better uncertainty estimates.
        """
        # Sample a batch of validation edges
        sample_size = min(1000, self.data.valid_edge_index.size(1))
        indices = torch.randperm(self.data.valid_edge_index.size(1))[:sample_size]
        
        sample_edge_index = self.data.valid_edge_index[:, indices]
        sample_edge_type = self.data.valid_edge_type[indices]
        
        # Get predictions from all models
        predictions = self.ensemble.forward_ensemble(
            self.data.edge_index,
            self.data.edge_type,
            sample_edge_index,
            sample_edge_type
        )
        
        preds_stack = torch.stack(predictions, dim=0)  # (num_models, num_samples)
        
        # Compute pairwise disagreement
        diversity = preds_stack.var(dim=0).mean().item()
        
        return diversity
    
    def save_checkpoint(self, epoch, name=None):
        """Save ensemble checkpoint."""
        checkpoint_dir = Path('saved_models/experiment_2')
        checkpoint_dir.mkdir(exist_ok=True)
        
        
        if name is None:
            name = f'{self.config.get_section("dataset")["name"]}_ensemble_checkpoint_epoch_{epoch}.pth'
           

        checkpoint_path = checkpoint_dir / name
        
        checkpoint = {
            'epoch': epoch,
            'num_models': self.ensemble.num_models,
            'encoder_args': self.ensemble.encoder_args,
            'decoder_args': self.ensemble.decoder_args,
            'models': [model.state_dict() for model in self.ensemble.models],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Ensemble checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path, load_optimizer=False):
        """Load ensemble checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        for i, state_dict in enumerate(checkpoint['models']):
            self.ensemble.models[i].load_state_dict(state_dict, strict=False)
            self.ensemble.models[i].eval()
        self.ensemble.eval()
        
        if load_optimizer:
            for i, state_dict in enumerate(checkpoint['optimizers']):
                self.optimizers[i].load_state_dict(state_dict)
        
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [], 'eval_metrics': [], 'ensemble_diversity': []
        })
        self.epoch = checkpoint['epoch']
        
        self.logger.info(f"Ensemble checkpoint loaded from {checkpoint_path}")

    def calibrate_pipeline(self, method, model, max_iters=50, lr=0.01):
        """Main entry point for calibration."""

        self.logger.info("Starting calibration process...")
        type_params = {
            'type': self.config.get_section('calibration').get('type', 'standard'),
            'mc_samples': self.config.get_section('calibration').get('mc_samples', 10)
        }
        self.logger.info(f"Uncertainty model type: {type_params}")
        if method == 'scalar':
            return self.calibrate_scalar_temperature(model, max_iters, lr, type_params)
        elif method == 'platt_scaling':
            return self.calibrate_platt_scaling_temperature(model, max_iters, lr, type_params)
        elif method == 'isotonic_regression':
            return self.calibrate_isotonic_regression(model)
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
    

    def calibrate_platt_scaling_temperature(self, model, max_iters=50, lr=0.01, type_params= None):
        
        self.logger.info(f"Calibration method: {self.config.get_section('calibration')['method']}")


        platt_model = PlattScalingEnsemble(mc_samples=type_params['mc_samples']).cuda()
        inference_params = {**type_params, 'return_logits': True, 'num_negatives': 1}
        pos,neg = self.inference(model, self.data.valid_edge_index, self.data.valid_edge_type, inference_params)
        pos = torch.stack(pos, dim=1).mean(dim=1) 
        neg = torch.stack(neg, dim=1).mean(dim=1) 
        platt_model.set_parameters(
                    pos_logits=pos,
                    neg_logits=neg,
                    lr=lr,
                    max_iters=max_iters
        )
        return platt_model
    def calibrate_scalar_temperature(self, model, max_iters=50, lr=0.01, type_params = None):
        """Calibrate scalar temperature parameter on validation set for ensemble.
        
        This is the traditional temperature scaling approach where a single
        scalar T divides all ensemble logits uniformly.
        
        Args:
            model: Ensemble model with scalar temperature parameter
            max_iters: Maximum optimization iterations
            lr: Learning rate for optimizer
            type_params: Additional parameters for calibration
            
        Returns:
            dict: Calibrated temperature value
        """
   
        calibrator = TemperatureEnsemble(init_temp=1.0).cuda()

        inference_params = {**type_params, 'return_logits': True, 'num_negatives': 1}
        pos,neg = self.inference(model, self.data.valid_edge_index, self.data.valid_edge_type, inference_params)

        pos = torch.stack(pos, dim=1).mean(dim=1) 
        neg = torch.stack(neg, dim=1).mean(dim=1)  
        
        calibrator.set_temperature(
            pos_logits=pos,
            neg_logits=neg,
            lr=lr,
            max_iters=max_iters
        )

        return calibrator
    
    def calibrate_isotonic_regression(self, model):
        """Calibrate using isotonic regression on validation set for ensemble.
        
        For ensembles, we fit isotonic regression on the ensemble mean predictions
        to calibrate the aggregated uncertainty estimates.
        
        Args:
            model: Ensemble model
            
        Returns:
            dict: Calibration results
        """


        self.logger.info("Starting Isotonic Regression Calibration for Ensemble")

        # Get ensemble predictions (raw logits) and labels
        model.eval()
        with torch.no_grad():
            out, labels, _ = self.inference(model, self.data.valid_edge_index, self.data.valid_edge_type, {'return_logits': False, 'num_negatives': 1})    

        probab = torch.sigmoid(out)
        probab = probab.cpu().numpy()
        labels = labels.cpu().numpy()

        iso_reg = IsotonicCalibrator(out_of_bounds='clip')
        iso_reg.fit(probab, labels)

        return iso_reg


    @torch.no_grad()
    def test_uncertainty(self,model, test_edge_index, test_edge_type, params):
        """
        Evaluate uncertainty on test set using ensemble variance.
        
        Returns uncertainty metrics: Brier score, ECE, etc.
        """
        params = {**params, 'num_negatives': 1, 'return_logits': False}
        y_out, y_true, y_variance = self.inference(model,test_edge_index, test_edge_type, params)
        y_true = y_true.detach().cpu().numpy()

        if 'calibration_model' in params and isinstance(params['calibration_model'], IsotonicCalibrator):

            calibrator = params['calibration_model']
            y_out = torch.sigmoid(y_out.detach()).cpu().numpy()
            y_prob_calib = calibrator.predict(y_out)
            self.logger.info("Uncertainty scores after calibration:")
            uncertainty_scores = compute_uncertainty(y_true, y_prob_calib)
            print(f"Variance of positives and negatives: {y_variance}")
        elif 'calibration_model' in params and isinstance(params['calibration_model'], PlattScalingEnsemble):
            calibrator = params['calibration_model'].eval()
            y_prob_calib = calibrator(y_out).detach().cpu().numpy()
            self.logger.info("Applied Platt Scaling with Ensemble calibration")
            uncertainty_scores = compute_uncertainty(y_true, y_prob_calib)
            print(f"Variance of positives and negatives: {y_variance}")

        elif 'calibration_model' in params and isinstance(params['calibration_model'], TemperatureEnsemble):
            calibrator = params['calibration_model'].eval()
            y_out_calib = calibrator(y_out.detach()).detach().cpu().numpy()
            print("Applied Scalar Temperature Scaling calibration")
            uncertainty_scores = compute_uncertainty(y_true, y_out_calib)
            print(f"Variance of positives and negatives: {y_variance}")
        else:
            y_prob = torch.sigmoid(y_out.detach()).cpu().numpy()
            # print(f"Calibrated probabilities (first 10): {y_prob[:10]}\n Uncalibrated probabilities (first 10): {y_out.detach().cpu().numpy()[:10]}")
            uncertainty_scores = compute_uncertainty(y_true, y_prob)
            print(f"Variance of positives and negatives: {y_variance}")



        for metric, value in uncertainty_scores.items():
            if isinstance(value, float):
                self.logger.info(f"{metric}: {value:.4f}")
            else:   
                self.logger.info(f"{metric}: {value}")
        
        return uncertainty_scores
    
    @torch.no_grad()
    def inference(self, model: DeepEnsemble, eval_edge_index, eval_edge_type, params):

        model.eval()
        neg_edge_index, neg_edge_type = negative_sampling(eval_edge_index, eval_edge_type, self.data.num_nodes, params.get('num_negatives', 1))
        
        combined_edge_index = torch.cat([eval_edge_index, neg_edge_index], dim=1)
        combined_edge_type = torch.cat([eval_edge_type, neg_edge_type])
        num_pos = eval_edge_index.size(1)

        z_all = model.encode(self.data.edge_index, self.data.edge_type)
        combined_out = model.decode(z_all, combined_edge_index, combined_edge_type) 
        
        positives = [out[:num_pos] for out in combined_out]
        negatives = [out[num_pos:] for out in combined_out]
        
        if params.get('return_logits', False): 
            return positives, negatives 
        
        out = torch.cat([torch.stack(positives, dim=0).mean(dim=0), torch.stack(negatives, dim=0).mean(dim=0)], dim=0)  # Shape: (num_samples, num_models)
        variance = torch.stack(positives, dim=0).var(dim=0).mean(), torch.stack(negatives, dim=0).var(dim=0).mean()  # Variance across ensemble members for positives and negatives

        gt = torch.cat([torch.full_like(eval_edge_type, 1), torch.full_like(neg_edge_type, 0)])

        return out, gt, variance
            

