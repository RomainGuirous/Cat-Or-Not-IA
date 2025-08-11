import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashfakyeafi/cat-dog-images-for-classification")

print("Path to dataset files:", path) 
#chemin de création: "C:\Users\Administrateur\.cache\kagglehub\datasets\"
# déplacer les bons fichiers dans data/ (cat_dog/ et cat_dog.csv)