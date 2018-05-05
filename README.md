# pytorch-SentimentAnalysis


## Description
This is a toy for learning pytorch for the first time. It includes the implementation of GRU/CNN for sentiment analysis.
Welcome raise issue if you have any questions about the code :)

Thanks to the code from https://github.com/akurniawan/pytorch-sentiment-analysis , I refer to his code for implementing the LSTM-based sentiment analysis.

## Dataset

I found the dataset from the repository of @akurniawan, it's a Twitter dataset. You can download the dataset from https://drive.google.com/file/d/1Go7FXn4mpIgle1X2mO1xYPDO0ZgWuPI6/view , which is simplified version of http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip .
However, the dataset itself was not really all human labeled, which means it's not accurate.If you have your own dataset, you had better use it.

## WordVector

You can choose to use pretrained wordvector to accerate training process and improve the accuracy. I use pretrained wordvector from GoogleNews-vectors-negative300, and there is a simplified version GoogleNews-vectors-negative300-SLIM. Just Google GoogleNews-vectors-negative300 then download the file. Of course you can use your own wordvector. 



