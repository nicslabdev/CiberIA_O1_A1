import numpy as np
from typing import Optional, Dict, Any

from modules import (
    DataCollectionModule,
    PreprocessingModule,
    AdversarialTrainingModule,
    MixedDriftHandlingModule,
    DriftCorrectionModule,
    AdaptiveModule,
    CompositeModelModule,
    ResultsModule
)

class ThreatDetectionSystem:
    """Sistema completo de detección de amenazas."""
    
    def __init__(self):
        """Inicializa todos los módulos del sistema."""
        self.data_collection = DataCollectionModule()
        self.preprocessing = PreprocessingModule()
        self.adversarial_training = AdversarialTrainingModule()
        self.drift_handling = MixedDriftHandlingModule()
        self.drift_correction = DriftCorrectionModule()
        self.adaptive = AdaptiveModule()
        self.model = CompositeModelModule()
        self.results = ResultsModule()
    
    def process_batch(self, 
                     dataset_path: str,
                     synthetic_data: Optional[np.ndarray] = None,
                     gan_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Procesa un batch completo de datos.
        
        Args:
            dataset_path: Ruta al dataset
            synthetic_data: Datos sintéticos
            gan_data: Datos de GAN
            
        Returns:
            Dict[str, Any]: Resultados del procesamiento
        """
        # 1. Recopilación de datos
        raw_data = self.data_collection.collect_data(dataset_path, synthetic_data, gan_data)

        print(raw_data)
        
        # 2. Preprocesamiento
        processed_data = self.preprocessing.preprocess(dataset_path, raw_data, apply_smote=True, apply_pca=True)

        print(processed_data)
        
        # 3. Entrenamiento adversario
        cleaned_data = self.adversarial_training.remove_adversarial_samples(processed_data)
        
        '''
        # 4. Detección de drift
        drift_detected, drift_type = self.drift_handling.detect_drift(
            cleaned_data,
            self.results.get_current_distribution()
        )
        
        if drift_detected:
            # 5. Corrección de drift
            corrected_data = self.drift_correction.correct_drift(cleaned_data, drift_type)
            # 6. Adaptación
            adapted_data = self.adaptive.adapt(corrected_data, None)  # TODO: Pasar modelo GAN
        else:
            adapted_data = cleaned_data
        '''
        
        # 7. Predicción y métricas
        predictions, metrics = self.model.predict(cleaned_data)
        
        # 8. Visualización de resultados
        self.results.display_results(predictions, metrics)
        
        return {
            'predictions': predictions,
            'metrics': metrics,
        } 
    

if __name__ == "__main__":
    system = ThreatDetectionSystem()
    system.process_batch(dataset_path="/mnt/AI-DATA/alara/CiberIA_O1/Data", synthetic_data=None, gan_data=None)