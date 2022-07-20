**Picture this**: you're an employee at Tesla. You finally managed to run "hello world" in Python. Not much of a stretch. Elon's running around like a madman, demanding FSD for yesterday and crashing the stock market with his tweets. You've somehow trained a neural network capable of telling the difference between a green traffic light and a watermelon, yet it still somehow can't tell which weighs more, a ton of bricks or your mother. Like any typical human, you go to Stack Overflow to find out that it is the latter (not news, just common sense and general consensus). As usual, not far from reality. Now you need to find a way to compile it to run in real-time on a computer in your neighbour's Tesla. You turn to Jensen and your so called friends (perfect tens, yet imaginary) at Nvidia to set up some deep learning acceleration scripts and guess what? There are no working examples with TensorFlow or PyTorch. Jensen shrugs shoulders when you realise that all they've been doing for the last two years is hand out GPUs to scalpers and every other asshole who decided to start mining shitcoins in their parents basement. You scrap the idea of going to AMD, since they were dumbfounded when they first heard the term "deep learning". Again, nothing new this day and age. Then you come to an idea even the retards at r/wallstreetbets would have shat themselves at: "Wait, I should just build the Nvidia deep learning drivers myself", but then scratch that when you realise that Nvida hasn't got the slightest clue what open source means. Not to worry, you miserable sack of shit, that's where I come in, by doing something none of you people could have come up even if it were fed to you with a silver spoon.

I surfed the internet, went through all of the broken-ass nonexistent fragments of things engineers call documentation and all sorts of porn. I've seen things that even I didn't know existed.

**And this is what I came up with**:  
Since the makers of TensorFlow thought that backward compatibility is nonessential, I created two separate folders with necessary scripts that assist you in taking a [Keras][3] or native [TensorFlow][4] model and converting it to a [TensorRT][2] engine which in turn can be used to run inference better, faster and more reliably than the great-wall-of-mexico-guy gets to run again for president.

As for [PyTorch][5], I'm amazed by their user friendliness, something which cannot be said about the Lizard man, who owns them, but who cares? As long as I don't have to create an account to use it, I'm fine.

Folders have simple names, **pytorch** is meant for **PyTorch** scripts, **TF** for **Tensorflow** scripts, notice the different versions once you figure out how to count to two.

My lawyers advised me to be polite, but I don't have the slightest idea of what any of that means, so here's the conclusion:  
This repository contains the code related to model quantisation with TensorRT for TensorFlow 1.X, 2.X (highly recommend migrating to [2.X][1] to avoid migraines) and PyTorch (to be created)  

These scripts work on Nvidia GPUs and Nvidia Jetsons, (GTX1060 6Gb, Jetson Xavier and Jetson Nano 2Gb and 4Gb tested)

These scripts have been tested with Python 3.6.9 and:  
* TensorFlow 1.14.0 [`./tf1`](tf1)
* Tensorflow 2.5.0 [`./tf2`](tf2)  
* PyTorch Soon™

[1]:https://www.tensorflow.org/guide/migrate/tf1_vs_tf2
[2]:https://github.com/NVIDIA/TensorRT
[3]:https://github.com/keras-team/keras
[4]:https://github.com/tensorflow/tensorflow
[5]:https://github.com/pytorch/pytorch
