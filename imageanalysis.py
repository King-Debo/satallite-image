import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for plotting and visualization
import cv2 # for image processing
import tensorflow as tf # for machine learning
import geopandas as gpd # for geospatial data integration

# The URL of NASA's Earthdata Search website
NASA_URL = "https://search.earthdata.nasa.gov/search"

# The names of the celestial bodies to analyze
CELESTIAL_BODIES = ["Moon", "Mars"]

# The names of the environmental changes to monitor
ENVIRONMENTAL_CHANGES = ["Deforestation", "Urban Expansion", "Land Cover Change"]

# The size of the images to process
IMAGE_SIZE = (256, 256)

# The number of classes for image classification
NUM_CLASSES = len(CELESTIAL_BODIES) + len(ENVIRONMENTAL_CHANGES)

# The batch size for training and testing the machine learning model
BATCH_SIZE = 32

# The number of epochs for training the machine learning model
EPOCHS = 10

# The learning rate for the machine learning model
LEARNING_RATE = 0.001

# The name of the machine learning model to save or load
MODEL_NAME = "satellite_image_analysis_ai.h5"

# A function to download satellite images from NASA's Earthdata Search website
def download_satellite_images(url, query):
  # Import the requests and BeautifulSoup libraries
  import requests
  from bs4 import BeautifulSoup

  # Create an empty list to store the satellite images
  images = []

  # Send a GET request to the URL with the query as a parameter
  response = requests.get(url, params={"q": query})

  # Check if the response status code is 200 (OK)
  if response.status_code == 200:
    # Parse the response content as HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all the <img> tags with class "thumbnail" in the HTML
    img_tags = soup.find_all("img", class_="thumbnail")

    # Loop over the img_tags
    for img_tag in img_tags:
      # Get the image source (src) attribute from the img_tag
      img_src = img_tag["src"]

      # Download the image from the img_src using requests
      img_response = requests.get(img_src)

      # Check if the image response status code is 200 (OK)
      if img_response.status_code == 200:
        # Append the image content to the images list
        images.append(img_response.content)

        # Print a message indicating that the image was downloaded successfully
        print(f"Downloaded image from {img_src}")

      else:
        # Print a message indicating that there was an error downloading the image
        print(f"Error downloading image from {img_src}")

  else:
    # Print a message indicating that there was an error accessing the website
    print(f"Error accessing {url}")

  # Return the images list
  return images

# A function to preprocess satellite images to enhance image quality and remove noise
def preprocess_satellite_images(images):
  # Import the cv2 library
  import cv2

  # Create an empty list to store the preprocessed satellite images
  preprocessed_images = []

  # Loop over the images
  for image in images:
    # Convert the image from bytes to numpy array
    image = np.frombuffer(image, dtype=np.uint8)

    # Decode the image using cv2
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Resize the image to the desired size using cv2
    image = cv2.resize(image, IMAGE_SIZE)

    # Crop the image to remove unwanted borders using numpy slicing
    image = image[10:-10, 10:-10]

    # Apply a median filter to remove noise using cv2
    image = cv2.medianBlur(image, 5)

    # Apply a threshold to binarize the image using cv2
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Append the preprocessed image to the preprocessed_images list
    preprocessed_images.append(image)

  # Return the preprocessed_images list
  return preprocessed_images

# A function to detect objects in satellite images using the machine learning model
def detect_objects_in_satellite_images(images, model):
  # Import the TensorFlow and cv2 libraries
  import tensorflow as tf
  import cv2

  # Create an empty list to store the bounding boxes and labels for the detected objects
  objects = []

  # Loop over the images
  for image in images:
    # Convert the image to a tensor and add a batch dimension using tf
    image_tensor = tf.expand_dims(tf.convert_to_tensor(image), axis=0)

    # Use the model to predict the bounding boxes and scores for the image using tf
    boxes, scores = model.predict(image_tensor)

    # Loop over the boxes and scores
    for box, score in zip(boxes[0], scores[0]):
      # Check if the score is above a certain threshold (e.g., 0.5)
      if score > 0.5:
        # Convert the box coordinates from normalized to pixel values using np
        box = np.multiply(box, [image.shape[0], image.shape[1], image.shape[0], image.shape[1]])

        # Round the box coordinates to integers using np
        box = np.round(box).astype(np.int32)

        # Draw a rectangle on the image using cv2
        cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)

        # Get the label for the object using the score index and a predefined list of classes
        label = CELESTIAL_BODIES + ENVIRONMENTAL_CHANGES[score.argmax()]

        # Draw the label on the image using cv2
        cv2.putText(image, label, (box[1], box[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Append the box and label to the objects list
        objects.append((box, label))

    # Print a message indicating that the object detection was done successfully
    print(f"Detected {len(objects)} objects in the image")

  # Return the objects list
  return objects

# A function to classify satellite images using the machine learning model
def classify_satellite_images(images, model):
  # Import the TensorFlow and cv2 libraries
  import tensorflow as tf
  import cv2

  # Create an empty list to store the predicted classes for the satellite images
  classes = []

  # Loop over the images
  for image in images:
    # Convert the image to a tensor and add a batch dimension using tf
    image_tensor = tf.expand_dims(tf.convert_to_tensor(image), axis=0)

    # Use the model to predict the class for the image using tf
    class_probabilities = model.predict(image_tensor)

    # Get the index of the highest probability in the class_probabilities array using np
    class_index = np.argmax(class_probabilities)

    # Get the name of the class using the class_index and a predefined list of classes
    class_name = CELESTIAL_BODIES + ENVIRONMENTAL_CHANGES[class_index]

    # Draw the class name on the image using cv2
    cv2.putText(image, class_name, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Append the class name to the classes list
    classes.append(class_name)

    # Print a message indicating that the image classification was done successfully
    print(f"Classified the image as {class_name}")

  # Return the classes list
  return classes

# A function to integrate geospatial data with satellite images for location-based analysis
def integrate_geospatial_data_with_satellite_images(images, geodata):
  # Import the geopandas and rasterio libraries
  import geopandas as gpd
  import rasterio

  # Create an empty list to store the satellite images with geospatial information overlayed on them
  integrated_images = []

  # Load the geospatial data file using gpd
  gdf = gpd.read_file(geodata)

  # Loop over the images
  for image in images:
    # Open the image as a raster using rasterio
    with rasterio.open(image) as src:
      # Get the image metadata, such as width, height, transform, and crs
      meta = src.meta

      # Reproject the geospatial data to match the image crs using gpd
      gdf = gdf.to_crs(meta["crs"])

      # Overlay the geospatial data on the image using rasterio
      out_image, out_transform = rasterio.mask.mask(src, gdf.geometry, crop=True)

      # Update the image metadata with the new transform and shape
      meta.update({"transform": out_transform, "height": out_image.shape[1], "width": out_image.shape[2]})

      # Save the integrated image to a temporary file using rasterio
      with rasterio.open("temp.tif", "w", **meta) as dst:
        dst.write(out_image)

      # Read the temporary file as an image using cv2
      integrated_image = cv2.imread("temp.tif")

      # Append the integrated image to the integrated_images list
      integrated_images.append(integrated_image)

  # Return the integrated_images list
  return integrated_images

# A function to plot and visualize satellite images and their analysis results
def plot_and_visualize_satellite_images_and_results(images, results):
  # Import the matplotlib and seaborn libraries
  import matplotlib.pyplot as plt
  import seaborn as sns

  # Set the style and color palette of the plots using sns
  sns.set(style="darkgrid", palette="muted")

  # Create a figure and a list of subplots using plt
  fig, axes = plt.subplots(nrows=len(images), ncols=2, figsize=(10, 10))

  # Loop over the images and results
  for i, (image, result) in enumerate(zip(images, results)):
    # Get the objects and classes from the result
    objects, classes = result

    # Plot the image on the left subplot using plt
    axes[i][0].imshow(image)
    axes[i][0].set_title(f"Satellite Image {i+1}")
    axes[i][0].set_axis_off()

    # Plot a histogram of the classes on the right subplot using plt
    axes[i][1].hist(classes, bins=NUM_CLASSES)
    axes[i][1].set_title(f"Class Distribution {i+1}")
    axes[i][1].set_xlabel("Class")
    axes[i][1].set_ylabel("Frequency")

  # Adjust the spacing between the subplots using plt
  plt.tight_layout()

  # Show the figure using plt
  plt.show()

