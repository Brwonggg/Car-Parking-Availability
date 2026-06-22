## Preview

## Intro 
This is a practice project that counts a rough estimate of empty parking spaces in an image using image processing and trained by a convolutional neural network(CNN).

Heavily inspired by ([this repo](https://github.com/FanchenBao/detect_empty_parking_spot))

Get the dataset ([here](https://www.kaggle.com/datasets/fanchenbao/parking-spots))

## Technologies
- Python

## Running the project
To run the project in your own local environment, follow these steps:

1. Clone the repository to your local machine
2. Run `pip install -r requirements.txt`
3. Ensure that you have updated variables base_path, empty_folder, occupied_folder, TEST_IMG inside of files train_step.py, test_step.py, main.py, image_loader, empty.py, data_sorter.py ,,,,,,,,,,,,,,,,,,
4. Run `python3 train_step.py` to start training the model
5. Once model has completed training, run `python3 main.py` and the image with empty and occupied parking lots will be displayed in a new window

## Features 
1. If coords.txt already exists, you can skip this section. If coords.txt does not exist, you will be prompted in the terminal to enter the number of columns in each row, followed by the number of rows in your bounding rectangle. The bounding rectangle is defined by the positioning of the dots you use to define the four corners(top left, top right, bottom left, bottom right in that specific sequence) of each parking lot segment(there should be no obstacles inside of your bounding rectangle). Do this for one segment and then escape this sequence by pressing 'q' on your keyboard when done with that segment, the coordinates of the individual parking lots will then be saved into a txt file. You will then be prompted to input either 'y' or 'n' to continue defining more segments or continue with the rest of the process respectively.
2. The need to define the parking lots is required only the first time. For subsequent runs, the model will use the same coordinates that have already been saved to the text file. This step is required because this project isn't capable of identifying empty/occupied parking spots on its own without the user pointing it in the right area, the user has to define the boundaries of where each parking spot is in the image for the model to check against. This is because the model was fed examples of individual empty and occupied parking lots, not the entire segment or carpark as a whole.
3. The model learns from the dataset where it recognises the features of empty and occupied parking lots, it then inspects each area within the bounding rectangle and checks it against the features that it has learnt to determine whether it is empty or occupied, highlighting an empty lot in green and an occupied one in red.

## Project Structure
image_loader.py - image loading + preprocessing

data_sorter.py - preprocessing + data augmentation

draw_bounds.py - draw bounding rectangles of parking lot sections and adds (x,y) coordinates into coords.txt

empty.py - checks whether area within the bounding rectangle drawn has a car

model.py - contains VGG Model class with frozen layers 

test_step.py - testing loop + test dataset

train_step.py - training loop + train dataset

## Reasoning Behind Certain Choices
In earlier iterations where I used greyscale, the model would differentiate an occupied parking lot from an empty one based on the contrast between the car and the lot itself, so a greater contrast meant that there was a car occupying the lot and a smaller one meant it was empty. I noticed that the model would almost always come to the wrong conclusion when it came to lighter-coloured cars because they have a more subtle contrast to the background as opposed to darker-coloured ones which is more prominent to the model. As such, the contrast caused by the lighter-coloured car matches that of an empty parking lot it learnt from the training data and it then thinks that that spot is empty when it's actually occupied by a lighter-coloured car. In the end, I opted not to use greyscale and solve this issue with lighter-coloured cars by adding more training examples of it into the dataset which I go into further detail in the Improvements section.

### Version 1(CNN from scratch) > Version 2(YOLO) > Version 3(ResNet) > Version 4(VGG)
Version 1 of the project used a CNN that I trained from scratch using a Kaggle dataset(different from the current one) but it had enough capacity to memorize roughly 6000 images, which is considered a small dataset relative to other models trained in image processing, there was not enough diverse data to learn generalizable visual concepts and features. It was memorising the data rather than actually learning the features to help it differentiate empty from occupied so when it was tested on data it had not seen before, it couldn't differentiate between the two correctly, showing signs of overfitting as seen by the training accuracy being significantly higher than the testing.

Version 2 used a YOLO model that was trained on hundreds of thousands of images so there was more than enough data, but the type of data that it was trained on didn't match the aerial top-down view of the image I was using. The YOLO model couldn't understand and learn features from datasets that varied so much from the testing image, so it performed even worse than the CNN. This shows that more data isn't always the solution, the type of data has to be matching your own specific situation.

Version 3 used a ResNet model which was trained on the ImageNet dataset and has millions of annotated images, including aerial top-down views, so this model seemed the most suitable to learn the features required to correctly identify empty and occupied parking lots from an aerial view. However, that large of a dataset would have downsides as the training process would take significantly more time, taking 30+ minutes just to train so it's because of that long of a training time that this model was also deemed unfeasible.

Version 4 leveraged transfer learning of a VGG model. After experimenting with a few values, I found that 23 frozen layers yielded the best reults because they used general features learnt from ImageNet and the remaining unfrozen layers were able to pick up on features trained via the dataset provided, specific to this problem which solved the issue of overfitting seen in version 1. Those unfrozen layers were also learning from the same dataset used in version 1 which were aerial top-down shots of individual parking lots which was what was lacking in the data in version 2. I also found that this VGG model worked faster than the ResNet model in version 3 despite the VGG one having more parameters, so that was also another thing I took into consideration when deciding on this model.

## Lessons Learnt
### General Syntax Mistakes
All other modules that contain specific functions should funnel into main.py and never import variables from main into other modules because it will cause an import error. Instead create instances/variables in the module itself and then pass it into the function as a parameter/argument.

Do not define variables inside functions, have them defined outside or every time that function is called, the variable is recreated and this is especially important when doing accumulation of values such as loss.

### Data
Data is the most important thing when it comes to training a model, choose your data properly first before writing code. The data used to train your model should be near identical to the testing image used so things like the angle of the parking lot, whether it's from an aerial top-down view, from the perspective of a car or from a wall-mounted camera. The details matter because the model trains itself on the data that it is given, it can't be given training data from a wall-mounted camera and be expected to reach a conclusion for an aerial image.

Ensure that you are using the best data available for your project's goals because the type of data that you use, influences the way that you write the code afterwards, the way you load your images, store variables, the different file paths. It's very tedious to edit all the code after realising that your data isn't a good fit. 

Your data should have a balanced ratio between classes, you should have the same number of images/training data for both empty and occupied so that your model is equally trained on both and doesn't favour one over the other. When I did confidence probability checks, I noticed that the values were sitting around 0.4-0.6 which means that the model wasn't confident in differentiating between empty and occupied lots.

If you're working with a model, constantly check what's the data type of the tensors that you are passing into it, ensure that the data type matches or is what is required or everything else breaks. Things like converting tensors to longs and floats so that it can be passed into the model. Similarly, keep track of their shapes and the device they're on as well.

### Model
added in thersholf 

optimizing harder for in-distribution test accuracy doesn't necessarily produce a model that generalizes better to a genuinely different distribution 

the empty probability doesnt have a large variance so the model cant actuallt tell confidently what is what , probabilities clustered around 0.5 regardless of true label"

more dropout, fewer trainable layers, weight decay — is the standard remedy for exactly this overfitting signature

architecture, regularization, transfer learning, class balancing, normalization, threshold calibration

## Limitations
This project isn't capable of identifying empty/occupied parking spots on its own. The user has to manually draw rectangles to define the boundaries of a parking spot for the model to then inspect each area within the bounding rectangle and check it against the features that it has learnt from being trained on the data.

When I attempted to use the YOLO model, it was also unable to detect the parking lot availability from an aerial view because a large majority of the data in the YOLO model is trained on angles that come from cars or wall-mounted cameras and rarely the top-down view shown in the example image. The training data of car parks from an aerial view were lacking and as such, the model was unable to learn the features and identify a car from said angle.

## How It Can Be Improved 
To resolve the issue of the model mistaking occupied parking lots for empty ones for lighter-coloured cars, you can add specifically more white coloured cars to your data and ensure that it's labelled to be occupied such that the model can use these as references and train the parameters to better improve its detection when it comes to these types of cars. Add more white/light-coloured cars and not just any coloured car because you want to address this issue specifically and adding more data that doesn't include these light colours doesn't actually help your model get any better at resolving this issue of identifying occupied lots with light-coloured cars in them.





