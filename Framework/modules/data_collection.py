import pandas as pd
import os
import numpy as np
from typing import Optional

class DataCollectionModule:
    """Módulo para recopilación de datos desde múltiples fuentes."""
    
    def __init__(self):
        self.raw_data = None
    
    def collect_data(self, 
                    dataset_path: str,
                    synthetic_data: Optional[np.ndarray] = None,
                    gan_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Recopila datos desde múltiples fuentes.
        
        Args:
            dataset_path: Ruta al dataset original
            synthetic_data: Datos sintéticos generados
            gan_data: Datos generados por GAN
            
        Returns:
            np.ndarray: Datos crudos combinados
        """

        print("Reading data...")
        self.raw_data = pd.read_csv(os.path.join(dataset_path, 'CIC-IDS2017.csv'))

        return self.raw_data