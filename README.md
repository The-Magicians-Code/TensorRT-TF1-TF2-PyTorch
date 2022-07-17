**Picture this**: you're an employee at Tesla. You finally managed to run "hello world" in Python. Not much of a stretch. Elon's running around like a madman, demanding FSD for yesterday and crashing the stock market with his tweets. You've somehow trained a neural network capable of telling the difference between a green traffic light and a watermelon, yet it still somehow can't tell which weighs more, a ton of bricks or your mother. Like any typical human, you go to Stack Overflow to find out that it is the latter (not news, just common sense and general consensus). As usual, not far from reality. Now you need to find a way to compile it to run in real-time on a computer in your neighbour's Tesla. You turn to Jensen and your so called friends (perfect tens, yet imaginary) at Nvidia to set up some deep learning acceleration scripts and guess what? There are no working examples with TensorFlow or PyTorch. Jensen shrugs shoulders when you realise that all they've been doing for the last two years is hand out GPUs to scalpers and every other asshole who decided to start mining shitcoins in their parents basement. You scrap the idea of going to AMD, since they were dumbfounded when they first heard the term "deep learning". Again, nothing new this day of age. Then you come to an idea even the retards at r/wallstreetbets would have shat themselves at: "Wait, I should just build the Nvidia deep learning drivers myself", but then scratch that when you realise that Nvida hasn't got the slightest clue what open source means. Not to worry, you miserable sack of shit, that's where I come in, by doing something none of you people could have come up even if it were fed to you with a golden spoon.

I surfed the internet, went through all of the broken-ass nonexistent fragments of things engineers call documentation and all sorts of porn. I've seen things that even I didn't know existed.

**And this is what I came up with**:  
Since the makers of TensorFlow thought that backward compatibility is nonessential, I created two separate folders with necessary scripts that assist you in taking a Keras or native TensorFlow model and converting it to a TensorRT engine which in turn can be used to run inference better, faster and more reliably than the great-wall-of-mexico-guy gets to run again for president.

Folders have simple names, **PyTorch** is meant for **PyTorch**, **TF** for **Tensorflow**, notice the different versions once you know how to count to two.

My lawer advised me to be polite, but I don't have the slightest idea of what that could mean, so here's the conclusion:  
This repository contains the code related to model quantisation with TensorRT for [TensorFlow 1.X][1], [2.X][2] and [PyTorch][3] (to be created)

[1]:tf1
[2]:tf2
[3]:pytorch
