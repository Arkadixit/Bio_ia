import os
import random
import shutil

# Chemins
testing_dir = "Data/Testing"
validation_dir = "Data/Validation"

# Créer le dossier Validation s'il n'existe pas
os.makedirs(validation_dir, exist_ok=True)

# Pour chaque classe dans le dossier Testing
for class_name in os.listdir(testing_dir):
    class_path = os.path.join(testing_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # Ignore les fichiers

    # Créer le sous-dossier correspondant dans Validation
    val_class_path = os.path.join(validation_dir, class_name)
    os.makedirs(val_class_path, exist_ok=True)

    # Liste des fichiers images
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    random.shuffle(images)

    # Calcul du nombre d’images à déplacer (15%)
    num_to_move = int(len(images) * 0.15)

    # Déplacement
    for image_name in images[:num_to_move]:
        src_path = os.path.join(class_path, image_name)
        dst_path = os.path.join(val_class_path, image_name)
        shutil.move(src_path, dst_path)

print("Déplacement terminé.")