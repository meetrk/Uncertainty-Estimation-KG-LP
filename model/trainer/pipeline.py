import numpy
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from model.trainer.basepipeline import BasePipeline
from utils.utils import negative_sampling
from utils.evaluation import compute_mrr,compute_uncertainty
from utils.utils import dropout_edges
from model.trainer.basepipeline import BasePipeline
from model.calibrator.tempscaling import TemperatureMCDropout, TemperatureScaling
from model.calibrator.plattscaling import PlattScaling, PlattScalingMCDropout
from model.calibrator.isotonic import IsotonicCalibrator
import torch

class Pipeline(BasePipeline):

    def __init__(self, model, data, config, logger):
        super().__init__(model, data, config, logger)
        

    def start_pipeline(self):
        max_epochs = self.train_config['epochs']
        eval_frequency = self.train_config.get('evaluation_frequency', 10)
        early_stopping = self.train_config['early stopping']['enabled']
        patience = self.train_config['early stopping'].get('patience', 10)
        delta = self.train_config['early stopping'].get('delta', 0.0)
        
        self.logger.info(f"Starting training for {max_epochs} epochs")
        best_val_mrr = -float('inf') 
        patience_counter = 0
        tqdm_range = range(1, max_epochs + 1)
        tqdm_range = tqdm(tqdm_range, desc="Training", unit="batch") 
        self.logger.info(f"Negative sampling ratio: {self.train_config['sampling']['negative_sampling_ratio']}")       

        for epoch in tqdm_range:
            
            loss = self.train()
            val_loss = 0
            print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
            self.writer.add_scalar('Loss/Train', loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)

            self.training_history['train_loss'].append({"epoch": epoch, "epoch_loss": loss})
            self.training_history['val_loss'].append({"epoch": epoch, "epoch_loss": val_loss})

            # Log gradients periodically
            if epoch % 10 == 0:  # Log gradients every 10 epochs
                self.log_model_gradients(epoch)


            # Evaluation
            if epoch % eval_frequency == 0:

                valid_scores, test_scores = self.test(test=self.train_config.get('test', True))

                current_val_mrr = valid_scores['mrr']
            
                # Check if current model is better than the best found so far
                if current_val_mrr - best_val_mrr > delta:
                    best_val_mrr = current_val_mrr
                    patience_counter = 0
                    
                    self.logger.info(f"New best model found! MRR: {current_val_mrr:.4f} > Previous: {best_val_mrr:.4f}")
                    if self.train_config['save_model']:
                        self.save_checkpoint(epoch) 
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                # Early Stopping Trigger
                if early_stopping and patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

                if test_scores is None:
                    test_scores = {"mrr": 0, "mean_rank": 0, "hits@1": 0, "hits@3": 0, "hits@10": 0}


                self.training_history['eval_metrics'].append({
                    "epoch": epoch,
                    "val_mrr": valid_scores["mrr"],
                    "val_mean_rank": valid_scores["mean_rank"],
                    "val_hits@1": valid_scores["hits@1"],
                    "val_hits@3": valid_scores["hits@3"],
                    "val_hits@10": valid_scores["hits@10"],
                    "test_mrr": test_scores["mrr"],
                    "test_mean_rank": test_scores["mean_rank"],
                    "test_hits@1": test_scores["hits@1"],  
                    "test_hits@3": test_scores["hits@3"],
                    "test_hits@10": test_scores["hits@10"],
                })
                self.logger.info(f"Epoch {epoch}: Val MRR = {valid_scores['mrr']:.4f}, Test MRR = {test_scores['mrr']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Mean Rank = {valid_scores['mean_rank']:.4f}, Test Mean Rank = {test_scores['mean_rank']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@1 = {valid_scores['hits@1']:.4f}, Test Hits@1 = {test_scores['hits@1']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@3 = {valid_scores['hits@3']:.4f}, Test Hits@3 = {test_scores['hits@3']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@10 = {valid_scores['hits@10']:.4f}, Test Hits@10 = {test_scores['hits@10']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val ECE= {valid_scores['ece']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val ACE= {valid_scores['ace']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Brier Score = {valid_scores['brier_score']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val True Probability = {valid_scores['prob_true']}")
                self.logger.info(f"Epoch {epoch}: Val Predicted Probability = {valid_scores['prob_pred']}")

                # Log evaluation metrics to TensorBoard
                self.writer.add_scalar('MRR/Validation', valid_scores['mrr'], epoch)
                self.writer.add_scalar('MRR/Test', test_scores['mrr'], epoch)
                self.writer.add_scalar('Mean_Rank/Validation', valid_scores['mean_rank'], epoch)
                self.writer.add_scalar('Mean_Rank/Test', test_scores['mean_rank'], epoch)
                self.writer.add_scalar('Hits@1/Validation', valid_scores['hits@1'], epoch)
                self.writer.add_scalar('Hits@1/Test', test_scores['hits@1'], epoch)
                self.writer.add_scalar('Hits@3/Validation', valid_scores['hits@3'], epoch)
                self.writer.add_scalar('Hits@3/Test', test_scores['hits@3'], epoch)
                self.writer.add_scalar('Hits@10/Validation', valid_scores['hits@10'], epoch)
                self.writer.add_scalar('Hits@10/Test', test_scores['hits@10'], epoch)
                self.writer.add_scalar('MC_Uncertainty/Brier_Score', valid_scores['brier_score'], epoch)
                self.writer.add_scalar('MC_Uncertainty/ECE', valid_scores['ece'], epoch)
                self.writer.add_scalar('MC_Uncertainty/ACE', valid_scores['ace'], epoch)

        
        # Close TensorBoard writer
        self.writer.close()
        self.logger.info("Training completed!")
        return self.training_history

    def load_pipeline(self, checkpoint_path, type,save):


        self.load_checkpoint(checkpoint_path)
        self.logger.info("Pipeline loaded from checkpoint.")
        uncertainty_samples = self.config.get_section('calibration')['mc_samples']
        scores = self.test_link_pred(
            type=type,
            model=self.model,
            valid_edge_index=self.data.test_edge_index,
            valid_edge_type=self.data.test_edge_type,
            mc_samples=uncertainty_samples)
        uncertainty_scores= self.test_uncertainty(
            model=self.model,
            test_edge_index=self.data.test_edge_index,
            test_edge_type=self.data.test_edge_type,
            params={'type': type, 'mc_samples': uncertainty_samples}
        )

        if self.config.get_section('calibration')['enabled']:
            self.logger.info("Starting calibration on test set...")
            calibration_model = self.calibrate_pipeline(
                method=self.config.get_section('calibration')['method'],
                model=self.model,
                max_iters=self.config.get_section('calibration').get('max_iters', 50),
                lr=self.config.get_section('calibration').get('learning_rate', 0.01)
            )
            scores = self.test_link_pred(
            type=type,
            model=self.model,
            valid_edge_index=self.data.test_edge_index,
            valid_edge_type=self.data.test_edge_type,
            mc_samples=uncertainty_samples,
            calibration_model=calibration_model
            )
            uncertainty_scores = self.test_uncertainty(
            model=self.model,
            test_edge_index=self.data.test_edge_index,
            test_edge_type=self.data.test_edge_type,
            params={'type': type, 'mc_samples': uncertainty_samples, 'calibration_model': calibration_model}
        )


        else:
            self.logger.info("Calibration not enabled; skipping calibration step.")

        if save:
            self.save_checkpoint(self.epoch, name=f'calibrated_{Path(checkpoint_path).name}')
        return uncertainty_scores

    def train(self):
        """
        Train the model on a single batch.
        """

        self.model.train()
        # Ensure input-dependent temperature is disabled during training
        if hasattr(self.model.decoder, 'use_calibration'):
            self.model.decoder.use_calibration = False
        
        self.optimizer.zero_grad()

        # dropout some edge randomly for training
        if self.train_config['sampling']['edge_dropout'] > 0:
            edge_index, edge_type = dropout_edges(self.data.edge_index, self.data.edge_type, self.train_config['sampling']['edge_dropout'])
        else:
            edge_index, edge_type = self.data.edge_index, self.data.edge_type

        z = self.model.encode(edge_index, edge_type)

        pos_out = self.model.decode(z, self.data.train_edge_index, self.data.train_edge_type)

        neg_edge_index,neg_edge_type = negative_sampling(self.data.train_edge_index, self.data.train_edge_type, self.data.num_nodes, self.train_config['sampling']['negative_sampling_ratio'])
        neg_out = self.model.decode(z, neg_edge_index, neg_edge_type)

        out = torch.cat([pos_out, neg_out])
        
        gt = torch.cat([torch.full_like(pos_out, self.train_config['label_smoothing']['positive']), torch.full_like(neg_out, self.train_config['label_smoothing']['negative'])])
        
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()

        loss = cross_entropy_loss + self.model_config['decoder']['l2_penalty'] * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()

        return float(loss)

    @torch.no_grad()
    def test(self, test = True):

        self.model.eval()
        valid_scores = compute_mrr(self.data.valid_edge_index, self.data.valid_edge_type,self.data, self.model, return_probs= False)
        if test:
            test_scores = compute_mrr(self.data.test_edge_index, self.data.test_edge_type,self.data, self.model, return_probs= False)
            return valid_scores, test_scores

        return valid_scores, None


    def calibrate_pipeline(self, method, model, max_iters=50, lr=0.01):
        """Main entry point for calibration."""

        self.logger.info("Starting calibration process...")
        type_params = {
            'type': self.config.get_section('calibration').get('type', 'standard'),
            'mc_samples': self.config.get_section('calibration').get('mc_samples', 10),
        }
        self.logger.info(f"Uncertainty model type: {type_params}")
        if method == 'scalar':
            return self.calibrate_scalar_temperature(model, max_iters, lr, type_params)
        elif method == 'platt_scaling':
            return self.calibrate_platt_scaling_temperature(model, max_iters, lr, type_params)
        elif method == 'isotonic_regression':
            return self.calibrate_isotonic_regression(model, type_params)
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
    

    def calibrate_platt_scaling_temperature(self, model, max_iters=50, lr=0.01, type_params= None):
        """Calibrate input-dependent temperature network on validation set.
        
        This method learns a small neural network that predicts per-query
        temperature values T(h,r) for each query, allowing the model to
        express uncertainty adaptively while preserving ranking.
        
        Args:
            model: GAE model with temp_network
            max_iters: Maximum optimization iterations
            lr: Learning rate for Adam optimizer
            
        Returns:
            dict: Final temperature statistics and loss

        """

        self.logger.info(f"Calibration method: {self.config.get_section('calibration')['method']}")
        if type_params['type'] == 'mc_dropout':
            platt_model = PlattScalingMCDropout(mc_samples=type_params['mc_samples'])
        else:
            platt_model = PlattScaling()

        inference_params = {**type_params, 'return_logits': True}
        pos,neg = self.inference(model, self.data.valid_edge_index, self.data.valid_edge_type, inference_params)
        if type_params['type'] == 'mc_dropout':
            pos = torch.stack(pos, dim=1) 
            neg = torch.stack(neg, dim=1)
        platt_model.set_parameters(
            pos_logits=pos,
            neg_logits=neg,
            lr=lr,
            max_iters=max_iters,
            lambda_mrl=self.config.get_section('calibration')['lambda']
        )
        return platt_model

    def calibrate_scalar_temperature(self, model, max_iters=50, lr=0.01, type_params = None):
        """Calibrate single scalar temperature parameter on validation set.
        
        This is the traditional temperature scaling approach where a single
        scalar T divides all logits uniformly.
        
        Args:
            model: GAE model with scalar temperature parameter
            max_iters: Maximum LBFGS iterations
            lr: Learning rate for LBFGS
            
        Returns:
            float: Calibrated temperature value
        """
        self.logger.info(f"Calibration method: {self.config.get_section('calibration')['method']}")
    

        if type_params['type'] == 'mc_dropout':
            calibrator = TemperatureMCDropout(init_temp=1.0)
        else:
            calibrator = TemperatureScaling(init_temp=1.0)

        inference_params = {**type_params, 'return_logits': True}
        pos,neg = self.inference(model, self.data.valid_edge_index, self.data.valid_edge_type, inference_params)

        if type_params['type'] == 'mc_dropout':
            pos = torch.stack(pos, dim=1)  
            neg = torch.stack(neg, dim=1)

        calibrator.set_temperature(
            pos_logits=pos,
            neg_logits=neg,
            lr=lr,
            max_iters=max_iters,
            lambda_mrl=self.config.get_section('calibration')['lambda']
        )

        return calibrator
    
    def calibrate_isotonic_regression(self, model, type_params):
        """Calibrate using isotonic regression on validation set.
        
        Args:
            model: GAE model
            
        Returns:
            dict: Calibration results
        """

        self.logger.info("Starting Isotonic Regression Calibration")

        with torch.no_grad():
            type_params = {**type_params, 'num_negatives': 1}
            if type_params['type'] == "mc_dropout":
                out, labels,_ = self.inference(model, self.data.valid_edge_index, self.data.valid_edge_type, type_params)     
            else:
                out, labels = self.inference(model, self.data.valid_edge_index, self.data.valid_edge_type, type_params)     
        
        probab = torch.sigmoid(out)
        probab = probab.cpu().numpy()
        labels = labels.cpu().numpy()
        if type_params['type'] == 'mc_dropout':
            probab = probab.mean(axis=1)
        iso_reg = IsotonicCalibrator(out_of_bounds='clip')
        iso_reg.fit(probab, labels)
        self.logger.info("Isotonic Regression model fitted.")
        self.logger.info("Calibration Complete!")

        return iso_reg

    @torch.no_grad()
    def inference(self, model,eval_edge_index, eval_edge_type, params):
        model.eval()
        if params['type'] == 'mc_dropout':
            out = []
            gt = []
            positives = []
            negatives =[]
            neg_edge_index,neg_edge_type = negative_sampling(eval_edge_index, eval_edge_type, self.data.num_nodes, params.get('num_negatives', 1))
            for _ in range(params['mc_samples']):
                model.encoder.mc_dropout = True
                z = model.encode(self.data.edge_index, self.data.edge_type)
                pos_out = model.decode(z, eval_edge_index, eval_edge_type)
                neg_out = model.decode(z, neg_edge_index, neg_edge_type)
                positives.append(pos_out)
                negatives.append(neg_out)
            if params.get('return_logits', False): 
                return positives, negatives
            out = torch.cat([torch.stack(positives, dim=1), torch.stack(negatives, dim=1)], dim=0) # Shape: (num_samples, mc_samples)
            variance = torch.stack(positives, dim=1).var(dim=1).mean(), torch.stack(negatives, dim=1).var(dim=1).mean()
            gt = torch.cat([torch.full_like(pos_out, 1), torch.full_like(neg_out, 0)])
            return out, gt, variance
        else:
            z = model.encode(self.data.edge_index, self.data.edge_type)
            pos_out = model.decode(z, eval_edge_index, eval_edge_type)
            neg_edge_index,neg_edge_type = negative_sampling(eval_edge_index, eval_edge_type, self.data.num_nodes, params.get('num_negatives', 1))
            neg_out = model.decode(z, neg_edge_index, neg_edge_type)  
            if params.get('return_logits', False):
                    return pos_out, neg_out
            out = torch.cat([pos_out, neg_out])
            gt = torch.cat([torch.full_like(pos_out, 1), torch.full_like(neg_out, 0)])    
        return out, gt
    
    def test_uncertainty(self, model,test_edge_index, test_edge_type, params):

        params = {**params, 'num_negatives': 1}
        if params['type'] == 'mc_dropout':
            y_out, y_true,y_variance = self.inference(model,test_edge_index, test_edge_type, params)
        else:
            y_out, y_true = self.inference(model,test_edge_index, test_edge_type, params)
        y_true = y_true.detach().cpu().numpy()

        if 'calibration_model' in params and isinstance(params['calibration_model'], IsotonicCalibrator):
            calibrator = params['calibration_model']
            y_out = torch.sigmoid(y_out.detach()).cpu().numpy()
            if params['type'] == 'mc_dropout':
                y_prob_calib = calibrator.predict(y_out.mean(axis=1))
            else:
                y_prob_calib = calibrator.predict(y_out)
            self.logger.info("Uncertainty scores after calibration:")
            uncertainty_scores = compute_uncertainty(y_true, y_prob_calib)
        elif 'calibration_model' in params and isinstance(params['calibration_model'], TemperatureScaling):
            calibrator = params['calibration_model'].eval()
            y_out_calib = calibrator(y_out.detach()).detach().cpu().numpy()
            y_prob_calib = torch.sigmoid(torch.tensor(y_out_calib)).numpy()
            uncertainty_scores = compute_uncertainty(y_true, y_prob_calib)
        elif 'calibration_model' in params and isinstance(params['calibration_model'], PlattScaling):
            calibrator = params['calibration_model'].eval()
            y_out_calib = calibrator(y_out.detach()).detach().cpu().numpy()
            y_prob_calib = torch.sigmoid(torch.tensor(y_out_calib)).numpy()
            uncertainty_scores = compute_uncertainty(y_true, y_prob_calib)
        elif 'calibration_model' in params and isinstance(params['calibration_model'], PlattScalingMCDropout):
            calibrator = params['calibration_model'].eval()
            y_prob_calib = calibrator(y_out).detach().cpu().numpy()
            self.logger.info("Applied Platt Scaling with MC Dropout calibration")
            uncertainty_scores = compute_uncertainty(y_true, y_prob_calib)
        elif 'calibration_model' in params and isinstance(params['calibration_model'], TemperatureMCDropout):
            calibrator = params['calibration_model'].eval()
            y_prob_calib = calibrator(y_out).detach().cpu().numpy()
            self.logger.info("Applied Temperature Scaling with MC Dropout calibration")
            uncertainty_scores = compute_uncertainty(y_true, y_prob_calib)
        else:
            if params['type'] == 'standard':
                y_prob = torch.sigmoid(y_out.detach()).cpu().numpy()
                uncertainty_scores = compute_uncertainty(y_true, y_prob)
            else:
                y_prob = torch.sigmoid(y_out.detach()).cpu().numpy()
                y_prob = y_prob.mean(axis=1)
                uncertainty_scores = compute_uncertainty(y_true, y_prob)
                print("Variance of MC Dropout predictions (positives):", y_variance)


        for metric, value in uncertainty_scores.items():
            if isinstance(value, float):
                self.logger.info(f"{metric}: {value:.4f}")
            else:   
                self.logger.info(f"{metric}: {value}")
        
        return uncertainty_scores