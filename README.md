## Preview(video demonstration)

## Intro 
This is a practice project that counts the number of empty parking spaces in an image, using object detection and training of a convolutional neural network to do so.

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

converting from dtype to match shape 

## Why I made certain choices/decisons 

Using cross entropy loss instead of bce 

test step is not required i just wanted to check the model training whether it was being effective

also note that opencv uses BGR and not RGB thats why my value for the red boxes is (0,0,255)

also have this issue where i was using greyscale before but then i realized that it was actually hurting the accuracy because of the contrast for lighter coloured cars is weaker and so the model cant tell




## Lessons I Learnt

what i learnt is that you shouldnt take long breaks duing projects because then it gets very hard to get back into the flow of what youre doing or if you do at least utilise comments and ur thought process

i had an issue of the data overfitting which means my training loss was much lower than the test one and accuracy much higher so i added data augmentation to increase the number of examples and it didnt do anything, tried decreasing the number of epochs and that didnt help either, tried to decrease the number of hidden layers and the thing that increased the test accuracy the most was dividing the X tensors by 255

trying to import main because you have variables in it wont work and will cause an import error, the way around this is to instead create instances/vraibalse in thta module itself and pass it in 

kept running into this error of conv2d layer receiving invalid combination of arguments 

data type matching is an important key so you have to first sort out your data properly if not everything else breaks like converting it to longs and floats 

I ran into this issue of the model either overcounting or under counting the number of empty lots and also over using the rectangles doing multiple for a single spot

there was this blue rectangle

dont put your training step inside of count_empty if not its training everytime you count a spot, training should happen before inference and this was prob the cause for my model to just be predicting everything to either be empty or occupied because it was never actually properly trained to identify the differences

dont put variables to be defined inside of functions, have them defined outside or else everytime that function is called, the varibale is being recreated for no reason 

another issue causing this discrepenacy was becuase i was feeding my training step my test data isntead so this means that it was only able to learn from 20% of the data and shuffle=True so it was memorising as opposed to learning and overfitting as a result

another issue was the data i used, the data just sucked so make sure you use the best available data 

make sure that you choose your data properly first before you start because if not it gets bery troublesome and chaotic to change all the vode after realising your data is the bottleneck

another issue was the order of my code because i ran into the issue of having the model run and give the number of empty spots without me ever drawing the rectangles because the cv.waitkey was being blocked by the model training

you also need to be aware of the contents of the coords.txt file because the issue of me having 32 empty parking lots when there was only 14 was prob caused by the old x y positioning carrying over 

## Limitations

This isn't capable of identifying empty parking spots on its own, you have to draw the rectangles to define what is a parking spot, a work around this would be to feed the model images with labels, depicting what is any empty parking lot and what isnt

very bad issues when it came to overfitting, training accuracy would be near 100% but test would be less than 50 

Or accept this project demonstrates the pipeline (drawing ROIs, training, inference) and document the domain gap as a known limitation rather than chasing accuracy on mismatched data

mismatched training/test distributions, how you identified it, what the fix would be

i noticed that it gets the wrong conclusion when it comes to the lighter coloured cars


## How it can be improved 

using a yolo model instead becuase its trained on millions of data as opposed to the hundreds that we've provided, more accurate and its already built in so that oyu dont have to spend time building the cnn yourself

more data, more variety, more data augementation 

overfitting play around with dropout rates and learning rate and number of epochs, adding more pooling layers but in doing so you also need to change the number of in_features in Linear

tried adding weight decay to adam, reducing filters and 

you can combat the model messing up the lighter coloured cars by addign more white coloured cars to your data to specifically combat it


(Unit tests)