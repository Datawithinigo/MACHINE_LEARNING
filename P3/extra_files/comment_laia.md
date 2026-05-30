Ok, from Assignment 3, I've been looking because we had, when doing Windows, we did a random split, which meant that you mixed all the bits at random and took 20% for the test. 

The problem is that due to the way Sliding Windows were being done, with adjacent windows, one window and the next shared up to 9 of the 10 bits. 
So what happens? that if one of those windows goes to the train and the other goes to the test because of the random split, then the model that has almost seen the entire sequence has only not seen a bit, so it was not like memorizing instead of generalizing, you know? 

And then, what we have done is make a temporary split, which is simply to make it cleaner. For each patient, 80% of the bits go to train and the remaining 20% ​​go to test. So, there is no overlap between the windows and we don't have this problem. Then, there is another thing, which is that we could also have done it in such a way that, I don't know how many patients there were, I think 14, since 11 go to the train and 3 to the test. 

Okay, I have tried it and it turns out terrible, but it could also be because there is little data, because the patients are quite different from each other, but it is true that clinically it seems the best option. 

I would put it as a limitation and that's it, to say or something future, whatever, to say this could be interesting, surely the metrics would go down because they go down 100% and that, let's take it into account. 
Let's see, what do you think and since the metrics are very good now, maybe what I said about class weight is not necessary. You already tell me.