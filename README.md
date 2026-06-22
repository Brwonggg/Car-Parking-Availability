## Preview
<img width="955" height="543" alt="Screenshot 2026-06-21 at 2 54 04 PM" src="https://github.com/user-attachments/assets/701f427f-30c5-4427-ab3f-d1f18ae318b9" />

## Intro 
This is a practice project that counts a rough estimate of empty parking spaces in an image using image processing and trained by a convolutional neural network(CNN).

Heavily inspired by this [repo](https://github.com/FanchenBao/detect_empty_parking_spot)

## Technologies
- Python

## Running the project
To run the project in your own local environment, follow these steps:

1. Clone the repository to your local machine
2. Download the parking spot dataset [here](https://www.kaggle.com/datasets/fanchenbao/parking-spots)
3. Place images into:
   - `data/spots/empty/` (empty spot images)
   - `data/spots/parked/` (occupied spot images)
4. Place your own test image at `data/test_images/test_parking.png`, or update `TEST_IMG` in `main.py` to point to your own image
5. Run `pip install -r requirements.txt` 
6. Run the code

## Features 
1. If coords.txt already exists, you can skip this section. If coords.txt does not exist, you will be prompted in the terminal to enter the number of columns in each row, followed by the number of rows in your bounding rectangle. The bounding rectangle is defined by the positioning of the dots you use to define the four corners(top left, top right, bottom left, bottom right in that specific sequence) of each parking lot segment(there should be no obstacles inside of your bounding rectangle). Do this for one segment and then escape this sequence by pressing 'q' on your keyboard when done, the coordinates of the individual parking lots will then be saved into a txt file. You will then be prompted to input either 'y' or 'n' to continue defining more segments or continue with the rest of the process respectively.
2. The need to define the parking lots is required only the first time. For subsequent runs, the model will use the same coordinates that have already been saved to the text file. This step is required because this project isn't capable of identifying empty/occupied parking spots on its own without the user pointing it in the right area, the user has to define the boundaries of where each parking spot is in the image for the model to check against. This is because the model was fed examples of individual empty and occupied parking lots, not the entire segment or carpark as a whole.
3. The model learns from the dataset where it recognises the features of empty and occupied parking lots, it then inspects each area within the bounding rectangle and checks it against the features that it has learnt to determine whether it is empty or occupied, highlighting an empty lot in green and an occupied one in red.

## Project Structure
image_loader.py - image loading + preprocessing

data_sorter.py - preprocessing + data type formatting

draw_bounds.py - draw bounding rectangles of parking lot sections and adds (x,y) coordinates into coords.txt

empty.py - functions used to check whether area within the bounding rectangle drawn has a car

model.py - VGG Model class with frozen backbone layers 

test_step.py - testing loop

train_step.py - training loop

## Reasoning Behind Certain Choices
In earlier iterations where I used greyscale, the model would differentiate an occupied parking lot from an empty one based on the contrast between the car and the lot itself, so a greater contrast meant that there was a car occupying the lot and a smaller one meant it was empty. I noticed that the model would almost always come to the wrong conclusion when it came to lighter-coloured cars because they have a more subtle contrast to the background as opposed to darker-coloured ones which is more prominent to the model. As such, the contrast caused by the lighter-coloured car matches that of an empty parking lot it learnt from the training data and it then thinks that that spot is empty when it's actually occupied by a lighter-coloured car. In the end, I opted not to use greyscale and solve this issue with lighter-coloured cars by adding more training examples of it into the dataset which I go into further detail in the Improvements section.

### Version 1(CNN from scratch) > Version 2(YOLO) > Version 3(ResNet) > Version 4(VGG)
Version 1 of the project used a CNN that was trained from scratch using a Kaggle dataset(different from the current one) but it had enough capacity to memorize roughly 6000 images, which is considered a small dataset relative to other models trained in image processing. There was not enough diverse data to learn generalizable visual concepts and features. It was memorising the data rather than actually learning the features to help it differentiate empty from occupied so when it was tested on data it had not seen before, it couldn't differentiate between the two correctly, showing signs of overfitting as seen by the training accuracy being significantly higher than the testing.

Version 2 used a YOLO model that was trained on hundreds of thousands of images so there was more than enough data, but the type of data that it was trained on didn't match the aerial top-down view of the image used for testing. The YOLO model couldn't understand and learn features from datasets that varied so much from the testing image, so it performed even worse than the CNN. This shows that more data isn't always the solution, the type of data has to be matching your own specific situation.

Version 3 used a ResNet model which was trained on the ImageNet dataset and has millions of annotated images, including aerial top-down views, so this model seemed the most suitable to learn the features required to correctly identify empty and occupied parking lots from an aerial view. However, that large of a dataset would have downsides as the training process would take significantly more time, taking 30+ minutes just to train so it's because of that long of a training time that this model was also deemed unfeasible.

Version 4 leveraged transfer learning of a VGG model. After experimenting with a few values, I found that 23 frozen layers yielded the best reults because they used general features learnt from ImageNet and the remaining unfrozen layers were able to pick up on features trained via the dataset provided, specific to this problem which solved the issue of overfitting seen in version 1. Those unfrozen layers were also learning from the same dataset used in version 1 which were aerial top-down shots of individual parking lots which was what was lacking in the data in version 2. I also found that this VGG model worked faster than the ResNet model in version 3 despite having more parameters, so that was also another thing I took into consideration when deciding on this model.

## Lessons Learnt
### General Syntax Mistakes
All other modules that contain specific functions should funnel into main.py and never import variables from main into other modules because it will cause an import error. Instead create instances/variables in the module itself and then pass it into the function as a parameter/argument.

### Data
Data is the most important thing when it comes to training a model, choose your data properly first before writing code. The data used to train your model should be near identical to the testing image used so things like the angle of the parking lot, whether it's from an aerial top-down view, from the perspective of a car or from a wall-mounted camera. The details matter because the model trains itself on the data that it is given, it can't be given training data from a wall-mounted camera and be expected to reach a conclusion for an aerial image.

Ensure that you are using the best data available for your project's goals because the type of data that you use, influences the way that you write the code afterwards, the way you load your images, store variables, the different file paths. It's very tedious to edit all the code after realising that your data isn't a good fit. 

Your data should have a balanced ratio between classes, you should have the same number of images/training data for both empty and occupied so that your model is equally trained on both and doesn't favour one over the other. When I did confidence probability checks, I noticed that the values were sitting around 0.4-0.6 which means that the model wasn't actually that confident in differentiating between empty and occupied lots.

Keep track of the data type of the tensors that you are passing into a model, ensure that the data types match or is what is required. Things like converting tensors to longs and floats so that it can be passed into the model. And also keep track of their shapes and the device they're on.

### Model
Threshold tuning did not result in any significant improvements because as mentioned earlier, the probabilities from the confidence probability checks had little standard deviation, they clustered around 0.5 regardless of true label. So the model wasn't very confident in differentiating which was which so moving the threshold value made little to no difference in accuracy.

Overfitting was the most common problem among all the models so with time, I implemented more regularization techniques in the form of dropout, weight decay, frozen backbone layers and early stopping to reduce it. I tested dropout values in the model and eventually decided on the middle ground value of 0.5 because even though a higher value would theoretically reduce overfitting, the dataset is so small that reducing the dropout rate even more would disable even more neurons which are needed to identify the features and doing so would result in the model favouring one of the classes over the other. The frozen backbone layers keep the general features that VGG is trained on while the unfrozen ones allow it to develop the ability to spot features specific to this situation.

## Limitations
This project isn't capable of identifying empty/occupied parking spots on its own. The user has to manually draw rectangles to define the boundaries of a parking spot for the model to then inspect each area within the bounding rectangle and check it against the features that it has learnt from being trained on the data.

Optimizing harder for train/test accuracy doesn't necessarily produce a model that generalizes better to a completely new test image, the model can only perform well when the training data and test image are similar in nature.

## How It Can Be Improved 
To resolve the issue of the model mistaking occupied parking lots for empty ones for lighter-coloured cars, you can add specifically more white coloured cars to your data and ensure that it's labelled to be occupied such that the model can use these as references and train the parameters to better improve its detection when it comes to these types of cars. Add more white/light-coloured cars and not just any coloured car because you want to address this issue specifically and adding more data that doesn't include these light colours doesn't actually help your model get any better at resolving this issue of identifying occupied lots with light-coloured cars in them.

Add a larger variety of data, different lighting, angles so that the model can be more well-equipped and able to generalise features instead of relying on specific ones. 

Extend the scope of the project by including obstacles so the grass patches between parking segments will be highlighted in blue. We are already using multi-class classification so adding one more class shouldn't prove to be very difficult.

You can cut out the need to manually define the boundaries of the parking lots by using a YOLO model, specifically one that is trained using visdrone which is a dataset of drone/aerial top-down images/videos that come with labels and annotations.





