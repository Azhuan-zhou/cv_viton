from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
import os
repo_path = "./local_model_dir"

model = SD3Transformer2DModel_Vton.from_pretrained(os.path.join(repo_path, "transformer_vton"))
model.save_pretrained("ckpt_dir", safe_serialization=True)
