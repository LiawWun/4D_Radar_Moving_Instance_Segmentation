import os
import torch

class TopKModelSaver:
    
    def __init__(self, save_dir, k=5):
        """
        Initialize the TopKModelSaver.

        Args:
            save_dir (str): Directory to save the top-k models.
            k (int): Number of top models to keep.
        """
        self.save_dir = save_dir
        self.k = k
        self.models = []  # List to store (validation_metric, file_name, epoch)
        os.makedirs(save_dir, exist_ok=True)

    def update(self, validation_metric, model, epoch):
        """
        Update the top-k models with the current model if it qualifies.

        Args:
            validation_metric (float): The metric to determine model quality (higher is better).
            model (torch.nn.Module): The model to potentially save.
            epoch (int): The current epoch number.
        """
        # If we have less than k models, simply add the current one
        if len(self.models) < self.k:
            file_name = f"top_{len(self.models) + 1}_epoch_{epoch}.pth"
            self._save_model(file_name, model)
            self.models.append((validation_metric, file_name, epoch))
            self.models = sorted(self.models, key=lambda x: x[0], reverse=True)

        # Otherwise, check if the current model is better than the worst one
        elif validation_metric > self.models[-1][0]:
            # Delete the lowest-ranked model file
            _, file_name_to_remove, _ = self.models.pop(-1)
            os.remove(os.path.join(self.save_dir, file_name_to_remove))

            # Add the new model
            file_name = f"top_{len(self.models) + 1}_epoch_{epoch}.pth"
            self._save_model(file_name, model)
            self.models.append((validation_metric, file_name, epoch))
            self.models = sorted(self.models, key=lambda x: x[0], reverse=True)

        # Rename the files to maintain rank order (top_1, top_2, ..., top_k)
        self._rename_files()

    def _save_model(self, file_name, model):
        """
        Save the model to disk.

        Args:
            file_name (str): The name of the file to save the model.
            model (torch.nn.Module): The model to save.
        """
        model_path = os.path.join(self.save_dir, file_name)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model with filename {file_name} at {model_path}")

    def _rename_files(self):
        """
        Rename the files to maintain proper ranking (top_1, top_2, ..., top_k).
        """
        for i, (metric, file_name, epoch) in enumerate(self.models):
            new_file_name = f"top_{i + 1}_epoch_{epoch}.pth"
            if file_name != new_file_name:
                os.rename(
                    os.path.join(self.save_dir, file_name),
                    os.path.join(self.save_dir, new_file_name),
                )
                self.models[i] = (metric, new_file_name, epoch)
