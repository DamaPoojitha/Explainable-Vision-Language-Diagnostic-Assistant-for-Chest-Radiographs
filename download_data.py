import kagglehub

# Indiana University Chest X-rays
iu_path = kagglehub.dataset_download("IndianaUniversity/chest-xray-images")
print("Indiana University dataset downloaded to:", iu_path)

# NIH Chest X-rays
nih_path = kagglehub.dataset_download("nih-chest-xrays/sample")
print("NIH Chest X-rays dataset downloaded to:", nih_path)
