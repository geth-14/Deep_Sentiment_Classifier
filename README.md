DESCRIPTION

This model is based upon Recurrent Neural Networks; it analyses the reviews aggregated from a product feedback dataset as against its ratings, thus mapping the sentiments to their corresponding scores. This model then behaves as a multi-class classifier being used to peruse the sentiments of newer reviews.

The Embedding layer is pretrained using the GloVe, and the model employs LSTM and Dropout for better results. 

CONSTITUENTS

1. RNN.py -> Script file
2. Sample_Data.csv -> Custom data file containing product reviews as against corresponding ratings on a scale of 5
3. Attributes.png -> Barplot of most occurring relevant attributes extracted from reviews for generating insights 
4. glove.6B.50d.txt -> 50-dimensional Global Vectors
