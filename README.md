## Preview(video demonstration)

## Intro 
This is a practice project that counts the number of empty parking spaces in an image using object detection and trained by a convolutional neural network(CNN).

## Technologies
- Python

## Running the project
To run the project in your own local environment, follow these steps:

1. Clone the repository to your local machine
2. Run `pip install -r requirements.txt`
3. Load in the datasets and sample image into train_data and img respectively <<<<>>>>
4. Run the code

## Features 
1. After running the code, you will be prompted to draw the bounding rectangles for every individual parking lot, both occupied and empty, in the GUI popup with your mouse. This project isn't capable of identifying empty/occupied parking spots on its own, the user manually draws the rectangles to define the boundaries of where each parking spot is in the image for the model to check.
2. The model will start training/learning the features and the time taken to do so will be displayed by the progress bar from the tqdm library along with the training loss and accuracy tracked every 3 epochs to show whether the training loop is training the model effectively. The overall trend should be an increase in accuracy and a decrease in loss.
3. The model learns from the dataset provided the features of empty and occupied parking lots, it then inspects each area within the bounding rectangle and checks it against the features that it has learnt. When the progress bar reaches 100% completion, a GUI with the number of empty lots will be displayed, additionally bounding the empty lots in green rectangles and the occupied ones in red.

## The process
image_loader.py - image loading + preprocessing

data_sorter.py - preprocessing + data augmentation

draw_rect.py - draw the rectangles in the GUI + passes the (x,y) coordinates into coords.txt

empty.py - checks whether area within the bounding rectangle drawn has a car

model.py - CNN model architecture

test_step.py - testing loop + test dataset

train_step.py - training loop + train dataset

## Why I Made Certain Choices 
In earlier iterations where I used greyscale, the model would differentiate an occupied parking lot from an empty one based on the contrast between the car and the lot itself, so a greater contrast meant that there was a car occupying the lot and a smaller one meant it was empty. I noticed that the model would almost always come to the wrong conclusion when it came to lighter-coloured cars because they have a more subtle contrast to the background as opposed to darker-coloured ones which is more prominent to the model. As such, the contrast caused by the lighter-coloured car matches that of an empty parking lot it learnt from the training data and it then thinks that that spot is empty when it's actually occupied by a lighter-coloured car. So in the end, I opted to not use greyscale to prevent this issue from occuring.

### Version 1(CNN) > Version 2(YOLO) > Version 3(ResNet) > Version 4(CNN)
Version 1 of the project: I used a convolutional neural network but it had enough capacity to memorize ~6000 images so there was not enough diverse data to learn generalizable visual concepts. It was memorising the data than actually learning the features to help it differentiate empty from occupied so when it was tested on data it had not seen before, it couldn't identify the two correctly. This is a case of overfitting.

Version 2: I used a YOLO model that was trained on hundreds of thousands of images so there was more than enough data, but the type of data that it was trained on didn't match the aerial top-down view of the image I was using. The YOLO model cannot understand and learn features from datasets that vary so much from the testing image, so it performed even worse than using the CNN.

Version 3: I used a ResNet model and I eventually stuck with this because it was trained on the ImageNet dataset which has millions of annotated images, including aerial top-down views, so this model seemed the most suitable to learn the features required to correctly identify empty and occupied parking lots from an aerial view.

Version 4: I went back to using a CNN but this time I was using a much larger dataset, went from 6000 images after data augmentation to 

## Lessons I Learnt

### General Syntax Mistakes
All other modules that contain specific functions should funnel into main.py and never import variables from main into other modules because it will cause an import error. Instead create instances/variables in the module itself and then pass it into the function as a parameter/argument.

Similarly, do not put training_step() inside of count_empty() because it will cause the model to restart training from scratch every time a spot is counted, so the training never actually happens and the model doesn't adjust the weights and biases on how to tell an empty parking lot from an occupied one. Training should happen before inference and this was an early reason behind why my model was predicting everything to either be empty or occupied and not actually using the learnings from its training loop.

Do not define variables inside functions, have them defined outside or every time that function is called, the variable is recreated and this is especially important when doing accumulation of values such as loss.

The model overcounted the number of empty lots and had multiple rectangles outlining a single spot. This was caused by the coords.txt file not being restarted every time I ran the code, so every subsequent run, it was using pre-existing coordinates and running it again. This explains why there were more than 14 recorded rectangles at a single time. The solution for this was by implementing close() every time before I drew the rectangle so that the txt file would start from scratch without any pre-existing coordinates.

### Data
Data is the most important thing when it comes to training a model, choose your data properly first before writing code. The data used to train your model should be near identical to the testing image used so things like the angle of the parking lot, whether it's from an aerial top-down view, from the perspective of a car or from a wall-mounted camera. The details matter because the model trains itself on the data that it is given, it can't be given training data from a wall-mounted camera and be expected to reach a conclusion for an aerial image.

Ensure that you are using the best data available for your project's goals because the type of data that you use, influences the way that you write the code afterwards, the way you load your images, store variables, the different file paths. It's just very tedious to edit all the code after realising that your data isn't a good fit. 

I spent hours trying to debug the code and find out what was wrong, only to find out that it was because the cars in the dataset were horizontal while the ones in my parking image were vertical and the fix was to just rotate the parking image with rotate() but that goes to show how close the training data has to be with the testing image.

If you're working with a model, constantly check what's the data type of the tensors that you are passing into it, ensure that the data type matches or is what is required or everything else breaks. Things like converting tensors to longs and floats so that it can be passed into the model. Similarly, keep track of their shapes and the device they're on as well.

Another careless mistake that caused an even bigger discrepenacy between the training and test accuracy was me feeding train_step() the test data instead of the training one so this meant that it was only able to learn from 20% of the data as denoted by the train_test_split ratio of 0.2 and shuffle=False so it was memorising the data as opposed to learning the features and overfitting the data as a result.

### Version 1(CNN) > Version 2(YOLO) > Version 3(ResNet)
In the CNN version of this project, I was playing around with the dropout rate in model.py and found out that reducing the dropout from 0.5 to 0 actually helped the model better differentiate occupied and empty lots which is counterintuitive because dropout is meant to reduce overfitting. This is because of the small size of the data so the model was already struggling to find features and reducing the number of parameters just worsened this so it resorted to just memorising rather than actually learning the features.

The change to the YOLO model was even worse and couldn't detect a car at all because it lacked training data from an aerial view, the angles that the images in the dataset and that of the testing image were very different, this ties back into how important it is for the model to be trained on the specific kind of data that you are using in testing.

Resnet was trained on millions of data and that's why it's better to utilise transfer learning and leverage models that other people have trained, rather than train your own model from scratch. I wish that I had this knowledge of using pretrained models through transfer learning so I didn't have to go through the struggle of training a CNN from scratch.

added in thersholf 

## Limitations
This project isn't capable of identifying empty/occupied parking spots on its own. The user has to manually draw the rectangles to define the boundaries of a parking spot for the model to then inspect each area within the bounding rectangle and check it against the features that it has learnt from being trained on the data.

When I attempted to use the YOLO model, it was also unable to detect the parking lot availability from an aerial view because a large majority of the data in the YOLO model is trained on angles that come from cars or wall-mounted cameras and rarely the top-down view shown in the example image. The training data of car parks from an aerial view were lacking and as such, the model was unable to learn the features and identify a car from said angle.
 
The CNN approach lacked sufficient data in general, having only around 6000 after data augmentation which proved to be insufficient because overfitting was still a prevalent problem. And overfitting would become the main bottleneck with every model I attempted, yolov8n, yolov8s(more parameters than yolov8n), CNN and even ResNet, as indicated by the training accuracy being near 100% within the first few epochs but the test accuracy always being 50% or lower.

## How It Can Be Improved 
The work around for this project being incapable of identifying empty/occupied parking spots on its own would be to use a YOLO model and feed the model entire parking lot images with labels, depicting with annotations what an empty and occupied lot looks like so that it can train itself on the parking lot as a whole. Right now, the dataset provided for training only shows what an individual empty or occupied parking lot looks like and so the model cross references this with the area inside of the rectangle to reach a conclusion.

In addition to that, if you're using the YOLO model, you can improve it by training the model yourself using VisDrone which is a dataset used to train models using aerial drone photography or by finding ideal weights that other people have tested on the Hugging Face website.

To resolve the issue of the model mistaking occupied parking lots for empty ones for lighter-coloured cars, you can add specifically more white coloured cars to your data and ensure that it's labelled to be occupied such that the model can use these as references and train the parameters to better improve its detection when it comes to these types of cars. Add more white/light-coloured cars and not just any coloured car because you want to address this issue specifically and adding more data that doesn't include these light colours doesn't actually help your model get any better at resolving this issue of identifying occupied lots with light-coloured cars in them.

For the CNN version, tune the hyperparameters and see what yields the best results. To reduce overfitting specifically, use a larger dataset, have more variety of data, use more data augementation to increase the overall number of testing examples that the model can learn from. Play around with the learning rate value, the number of epochs, add a weight decay to the optimizer or adjust the train_test_split ratio that you allocate. Tweak the dropout values, reduce filters/hidden layers or try adding more pooling layers, which reduces the number of parameters and how much it can memorise, but be mindful of adjusting the number of in_features in nn.Linear() in the classifier layer afterwards. Then use TensorBoard to test your hyperparameter tuning, visualise your findings and find which conditions are the best.




