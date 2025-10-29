import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """
    A YAML configuration file loader that parses and returns main sections.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the ConfigLoader with a configuration file path.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config_data = None
        
    def load(self) -> Dict[str, Any]:
        """
        Load and parse the YAML configuration file.
        
        Returns:
            Dictionary containing all configuration sections
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                self.config_data = yaml.safe_load(file)
            return self.config_data
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")
    
    def get_sections(self) -> Dict[str, Any]:
        """
        Get all main sections from the configuration.
        
        Returns:
            Dictionary with all top-level sections
        """
        if self.config_data is None:
            self.load()
        return self.config_data or {}
    
    def get_section(self, section_name: str) -> Any:
        """
        Get a specific section from the configuration.
        
        Args:
            section_name: Name of the section to retrieve
            
        Returns:
            The requested section data
            
        Raises:
            KeyError: If section doesn't exist
        """
        if self.config_data is None:
            self.load()
        
        if self.config_data is None or section_name not in self.config_data:
            raise KeyError(f"Section '{section_name}' not found in configuration")
        
        return self.config_data[section_name]
