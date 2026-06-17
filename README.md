## Preview(video demonstration)

## Intro 
This is a practice project that counts the number of empty parking spaces in an image, 

## Technologies(tech stack used)
- Python

## Running the project(pip install)
To run the project in your own local environment, follow these steps:

1. Clone the repository to your local machine
2. Run `pip install -r requirements.txt`
3. Load the dataset 
4. Copy the file path and paste it into the variable FILE_PATH in main.py
5. Run the code

## Features 
You give the data, and then when you run the code, there will be a progress bar using tqdm along with the training loss and accuracy tracked every 3 epochs to show whether the training loop is effective, the model will then become trained at identifying whether a selected spot is empty or occupied by a car, you would then have a gui pop up that allows you to draw the rectangles of each and every parking lot for the model to inspect and determine whether its empty or occupied, the output after you quit out of the application will then display the total number of empty parking lots 

## The process (How I built it)

another issue was the order of my code because i ran into the issue of having the model run and give the number of empty spots without me ever drawing the rectangles because the cv.waitkey was being blocked by the model training

## Why I made certain choices/decisons 


Using cross entropy loss instead of bce 

In earlier iterations where I used greyscale so the model would identify whether a parking lot was empty or occupied based on the contrast, I noticed that it will mostly always come to the wrong conclusion when it came to the lighter-coloured cars because a white car would have a more subtle contrast to the background as opposed to a black one which is more prominent to the model and as such, the contrast caused by the white car matches that of an empty parking lot it learnt from the training data, it then thinks that that spot is empty when it's actually occupied by a white car.

I tried training a cnn by myself, tried yolo both didnt work now im trying resnet
it had enough capacity to memorize 4872 images perfectly, but not enough diverse data to learn generalizable visual concepts.
but it's pretrained specifically for object detection at typical photographed angles — street level, dashcams, surveillance cameras at an angle.

## Lessons I Learnt
This project while not near perfect, has taught me a lot that I can definitely take away with me to become a better programmer.

The first of which is a more non-technical and general lesson which is to not take long breaks during projects because it then becomes very hard to get back into the flow of what you were doing or to pick up from where you left off of. Consistency is key but if you have to take an extended break, make sure to utilise comments and document your thought process at the time so that you have a sense of where to continue from.

### General Coding Findings
All of your other modules that contain functions should funnel into main and you should never be importing variables from main into other modules because it will cause an import error. The way around this is to instead create instances/variables in the module itself and then pass it into the function as a parameter/argument.

Similarly, don't put your training_step() inside of count_empty() because it will then cause the model to restart training every time a spot is counted so the training never actually happens and the model doesn't learn the weights and biases on how to tell an empty parking lot from an occupied one. Training should happen before inference and this was most likely the cause for my model predicting everything to either be empty or occupied and not actually using the learnings from its training loop because it was never actually properly trained to identify the differences. 

Also do not define variables inside functions, have them defined outside or every time that function is called, the variable is recreated and this is especially bad when you are doing accumulation of something.

### ML/DL Findings
Data is the most important thing when it comes to training a model. You have to ensure that the data type matches so first sort out data properly, if not everything else breaks. Things like converting tensors to longs and floats so that it can be passed into the model.

Adding on to that, make sure you choose your data properly first before you start, make sure that it's the kind of data that 

#### Version 1: Convolutional Neural Network
In the CNN version of this project, I often had the issue of my training accuracy being significantly higher than my testing accuracy which indicates that my model is overfitting the data so I tried adding more data for the model to extract features from through data augmentation but there was no noticeable improvements. I also tried decreasing the number of epochs and the number of hidden layers such that the model would be forced to generalize more instead of just memorising but those changes also had no effect on the test accuracy.

I also ran into the issue of the model overcounting the number of empty lots and having multiple rectangles outlining a single spot. This was caused by the coords.txt file not being restarted every time I ran the code, so every subsequent time I would run the code, it was using pre-existing coordinates and running it again that's why there were more than 14 recorded rectangles at a single time. The solution for this was by implementing close() every time before I drew the rectangle so that the text file would start from scratch without any pre-existing coordinates.

Another careless but mistake causing this discrepenacy between accuracy was becuase i was feeding my training step my test data isntead so this means that it was only able to learn from 20% of the data and shuffle=True so it was memorising as opposed to learning and overfitting as a result

I was playing around with the dropout rate in model.py and found out that reducing the dropout from 0.5 to 0 actually helped the model better detect which lots were occupied and which were empty which is counterintuitive because dropout is meant to reduce overfitting. I found out that this is because given the small size of the data 

tuning hyper parameters by playing with the values, i saw an increase in accuracy by chaning the ratio of my train_test_split from 0.2 to 0.4

the data is even more important than you realise, i spent hours trying to debug the code and find out whats wrong only to find out that its because the cars in the data are horiziontal while the parking image i am using, they are vertical, the fix to this was just rotating my own testing 

## Limitations
This project isn't capable of identifying empty/occupied parking spots on its own. The user has to manually draw the rectangles to define the boundaries of a parking spot for the model to then inspect each area within the bounding rectangle and check it against the features that it has learnt from being trained on the data.

When I attempted to use the YOLO model, it was also unable to detect the parking lot availability from an aerial view because a large majority of the data in the YOLO model is trained on angles that come from cars or wall-mounted cameras and rarely the top-down view shown in the example image. The training data of car parks from an aerial view were lacking and as such, the model was unable to learn the features and identify a car from said angle.
 
The CNN approach lacked sufficient data in general, having only around 6000 after data augmentation which proved to be insufficient because overfitting was still a prevalent problem. And overfitting would become the main bottleneck with every model I attempted, yolov8n.pt, yolov8s.pt(more parameters), CNN and even ResNet, as indicated by the training accuracy being near 100% within the first few epochs but the test accuracy always being 50% or lower.

## How It Can Be Improved 
The work around for this project being incapable of identifying empty/occupied parking spots on its own would be to use a YOLO model and feed the model whole parking lot images with labels, depicting with annotations what an empty and occupied lot looks like so that it can train itself on the parking lot as a whole. Right now, the dataset provided for training only shows what an individual empty or occupied parking lot looks like and so the model cross references this with the area inside of the rectangle to reach a conclusion.

In addition to that, if you're using the YOLO model, you can improve it by training the model yourself using VisDrone which is a dataset used to train models using aerial drone photography or by finding ideal weights that other people have tested on the Hugging Face website.

To combat the model messing up when it comes to lighter-coloured cars, you can add specifically more white coloured cars to your data and ensure that it's labelled to be occupied such that the model can use these as references and train the parameters to better improve its detection when it comes to these types of cars. Add more white/light-coloured cars and not just any coloured car because you want to address this issue specifically and adding more data that doesn't include these light colours doesn't actually help your model get any better at resolving this issue of identifying occupied lots with light-coloured cars in them.

Tune the hyperparameters and see what yields the best results. To reduce overfitting specifically, use a larger dataset, have more variety of data, use more data augementation to increase the overall number of testing examples that the model can learn from. Play around with the learning rate value, the number of epochs, add a weight decay to the optimizer or adjust the train_test_split ratio that you allocate. In CNNs, tweak the dropout values, reduce filters/hidden layers or try adding more pooling layers, which reduces the number of parameters and how much it can memorise, but be mindful of adjusting the number of in_features in nn.Linear() in the classifier layer afterwards. Then use TensorBoard to test your hyperparameter tuning, visualise your findings and find which conditions are the best.




