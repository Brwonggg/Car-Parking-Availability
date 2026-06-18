## Preview(video demonstration)

## Intro 
This is a practice project that counts the number of empty parking spaces in an image, using a ResNet model.

## Technologies
- Python

## Running the project
To run the project in your own local environment, follow these steps:

1. Clone the repository to your local machine
2. Run `pip install -r requirements.txt`
3. Load in the datasets and sample image into train_data and img respectively
4. Run the code

## Features 
1. After running the code, you will be prompted to draw the bounding rectangles for every individual parking lot, both occupied and empty, in the GUI with your mouse. This project isn't capable of identifying empty/occupied parking spots on its own, the user manually draws the rectangles to define the boundaries of where each parking spot is in the image for the model to check.
2. Wait 8-10 minutes for the model to train which will be displayed by the progress bar from the tqdm library along with the training loss and accuracy tracked every 3 epochs to show whether the training loop is training the model effectively. The overall trend should be an increase in accuracy and a decrease in loss.
3. The model learns from the datasets provided the features of empty and occupied parking lots, it then inspects each area within the bounding rectangle and checks it against the features that it has learnt. When the progress bar reaches 100% completion, a GUI with the number of empty lots will be displayed, additionally bounding the empty lots in green rectangles and the occupied ones in red.

## The process (How I built it)

image_loader.py

data_sorter.py

draw_rect.py

empty.py

model.py

train_step.py - tracks the time taken for the model to train. shows a progress bar, 

test_step.py

main.py

## Why I made certain choices/decisons 

even tried using different images for the bounding and there was a major difference

also changed the dataset simulatneously

Using cross entropy loss instead of bce 

In earlier iterations where I used greyscale so the model would identify whether a parking lot was empty or occupied based on the contrast, I noticed that it will mostly always come to the wrong conclusion when it came to the lighter-coloured cars because a white car would have a more subtle contrast to the background as opposed to a black one which is more prominent to the model and as such, the contrast caused by the white car matches that of an empty parking lot it learnt from the training data, it then thinks that that spot is empty when it's actually occupied by a white car.

Version 1 of the project had me using a convolutional neural network , it had enough capacity to memorize 4872 images perfectly, but not enough diverse data to learn generalizable visual concepts.

Version 2 was using a YOLO model it had enough capacity to memorize 4872 images perfectly, but not enough diverse data to learn generalizable visual concepts.

Version 3 was using a ResNet model and I eventually stuck with using this

## Lessons I Learnt
The first of which is a more general lesson but it's to not take long breaks during projects because it then becomes very hard to get back into the flow of what you were doing or to pick up from where you left off of. Consistency is key but if you have to take an extended break, make sure to utilise comments and document your thought process at the time so that you have a sense of where to continue from.

### General Syntax Mistakes
All other modules that contain specific functions should funnel into main.py and never import variables from main into other modules because it will cause an import error. The way around this is to instead create instances/variables in the module itself and then pass it into the function as a parameter/argument.

Similarly, do not put training_step() inside of count_empty() because it will then cause the model to restart training from scratch every time a spot is counted so the training never actually happens and the model doesn't adjust the weights and biases on how to tell an empty parking lot from an occupied one. Training should happen before inference and this was most likely the cause for my model predicting everything to either be empty or occupied and not actually using the learnings from its training loop because it was never actually properly trained to identify the differences. 

Do not define variables inside functions, have them defined outside or every time that function is called, the variable is recreated and this is especially bad when you are doing accumulation of values such as loss.

I also ran into the issue of the model overcounting the number of empty lots and having multiple rectangles outlining a single spot. This was caused by the coords.txt file not being restarted every time I ran the code, so every subsequent time I would run the code, it was using pre-existing coordinates and running it again that's why there were more than 14 recorded rectangles at a single time. The solution for this was by implementing close() every time before I drew the rectangle so that the text file would start from scratch without any pre-existing coordinates.

### Data
Data is the most important thing when it comes to training a model, make sure you choose your data properly first before you start. The data your using to train your model should be near identical to the image that you are trying to test it so things like the angle of the parking lot whether it's from an aerial top-down view, from the perspective of a car or from a wall-mounted camera. These kind of things matter because the model trains itself on the data that it is given, it can't be given training data from a wall-mounted camera and be expected to reach conclusions for an aerial 


You have to ensure that the data type matches so first sort out data properly, if not everything else breaks. Things like converting tensors to longs and floats so that it can be passed into the model.

Adding on to that, 

the data is even more important than you realise, i spent hours trying to debug the code and find out whats wrong only to find out that its because the cars in the data are horiziontal while the parking image i am using, they are vertical, the fix to this was just rotating my own testing 

Another careless but mistake causing this discrepenacy between accuracy was becuase i was feeding my training step my test data isntead so this means that it was only able to learn from 20% of the data and shuffle=True so it was memorising as opposed to learning and overfitting as a result

### Version 1(CNN) > Version 2(YOLO) > Version 3(ResNet)
In the CNN version of this project, I often had the issue of my training accuracy being significantly higher than my testing accuracy which indicates that my model is overfitting the data so I tried adding more data for the model to extract features from through data augmentation but there was no noticeable improvements. I also tried decreasing the number of epochs and the number of hidden layers such that the model would be forced to generalize more instead of just memorising but those changes also had no effect on the test accuracy.

I was playing around with the dropout rate in model.py and found out that reducing the dropout from 0.5 to 0 actually helped the model better detect which lots were occupied and which were empty which is counterintuitive because dropout is meant to reduce overfitting. I found out that this is because given the small size of the data 

## Limitations
This project isn't capable of identifying empty/occupied parking spots on its own. The user has to manually draw the rectangles to define the boundaries of a parking spot for the model to then inspect each area within the bounding rectangle and check it against the features that it has learnt from being trained on the data.

When I attempted to use the YOLO model, it was also unable to detect the parking lot availability from an aerial view because a large majority of the data in the YOLO model is trained on angles that come from cars or wall-mounted cameras and rarely the top-down view shown in the example image. The training data of car parks from an aerial view were lacking and as such, the model was unable to learn the features and identify a car from said angle.
 
The CNN approach lacked sufficient data in general, having only around 6000 after data augmentation which proved to be insufficient because overfitting was still a prevalent problem. And overfitting would become the main bottleneck with every model I attempted, yolov8n, yolov8s(more parameters than yolov8n), CNN and even ResNet, as indicated by the training accuracy being near 100% within the first few epochs but the test accuracy always being 50% or lower.

## How It Can Be Improved 
The work around for this project being incapable of identifying empty/occupied parking spots on its own would be to use a YOLO model and feed the model entire parking lot images with labels, depicting with annotations what an empty and occupied lot looks like so that it can train itself on the parking lot as a whole. Right now, the dataset provided for training only shows what an individual empty or occupied parking lot looks like and so the model cross references this with the area inside of the rectangle to reach a conclusion.

In addition to that, if you're using the YOLO model, you can improve it by training the model yourself using VisDrone which is a dataset used to train models using aerial drone photography or by finding ideal weights that other people have tested on the Hugging Face website.

To resolve the issue of the model mistaking occupied parking lots for empty ones for lighter-coloured cars, you can add specifically more white coloured cars to your data and ensure that it's labelled to be occupied such that the model can use these as references and train the parameters to better improve its detection when it comes to these types of cars. Add more white/light-coloured cars and not just any coloured car because you want to address this issue specifically and adding more data that doesn't include these light colours doesn't actually help your model get any better at resolving this issue of identifying occupied lots with light-coloured cars in them.

For the CNN version, tune the hyperparameters and see what yields the best results. To reduce overfitting specifically, use a larger dataset, have more variety of data, use more data augementation to increase the overall number of testing examples that the model can learn from. Play around with the learning rate value, the number of epochs, add a weight decay to the optimizer or adjust the train_test_split ratio that you allocate. In CNNs, tweak the dropout values, reduce filters/hidden layers or try adding more pooling layers, which reduces the number of parameters and how much it can memorise, but be mindful of adjusting the number of in_features in nn.Linear() in the classifier layer afterwards. Then use TensorBoard to test your hyperparameter tuning, visualise your findings and find which conditions are the best.




