# Weekly Schedule

### Week 1:
- Goal:
    - Obtain SEMG acquisition hardware
    - Begin research on neural networks for classification
- Completed:
    - Set up Myo armband from lab inventory
    - Found and read papers that may provide inspiration
- Next steps:
    - Obtain raw emg from Myo armband
    - Finalize scope/direction of project

### Week 2:
- Goal:
    - Obtain raw emg from Myo armband
    - Settle on shallow or deep learning
- Completed:
    - Recorded and plotted raw emg with Myo armband
    - Settled on Deep learning with convolutional neural networks
- Next steps:
    - Learn Pytorch
    - Find database for pretraining data

### Week 3:
- Goal:
    - Start learning pytorch
    - Get more data
- Completed:
    - Used pytorch tutorials to gain basic understanding of pytorch
    - Got data from Ninapro database
- Next steps:
    - Begin coding network architecture and training in pytorch

### Week 4:
- Goal:
    - Set up CNN architecture and windowing of dataset
    - Better understand pytorch API
- Completed:
    - Wrote classes for two CNN architectures (basic and enhanced) based on paper
    - Set up windowing pipeline for sampling the dataset
    - 260ms window with 235 ms overlap
- Next steps:
    - Debug to be able to feed data set into CNN to begin training


### Week 5:
- Goal:
    - Be able to feed data set into CNN to begin training
- Completed:
    - Figured out inputs/outputs for each layer (on both networks) in order to pass data set through network
    - Expanded data set to include all 10 subjects (Ninapro DB5)
    - Randomly split data set roughly 80-20 for training and validation
    - Only included samples corresponding to finger flexion gestures and neutral hand position (rest)
    - Began training network
    - Best test set accuracy approx. 50% for 6 classes
- Next Steps:
    - Improve accuracy
    - Look into using GPU


### Week 6:
- Goal:
    - Increase classification accuracy
    - Try training on GPU
- Completed:
    - Best training accuracy 98%, validation accuracy 85%
    - Added dataloaders to improve input of data (Easier setup of batch size, shuffling)
    - Concerns about overfitting
    - Collected EMG data from myself
- Next steps:
    - Systematic hyperparameter tuning on GPU
    - Setup transfer learning pipeline to begin training with my data


### Week 7:
- Goal
    - Tune hyperparameters and make sure there is no overfitting
    - Use transfer learning to incorporate my data
- Completed
    - Tuned most hyperparameters
    - Left to tune: Additional parameters for optimizer(momentum, weight decay)
    - Began transfer learning pipeline using fine tuning method
    - 90% test accuracy with my personal data
    - Finetuning seems to makes predictive model specific to one person. More experimentation is needed
    - Started script for real time prediction with finetuned model
- Next steps
    - Continue to explore transfer learning and improving accuracy
    - Improve real time prediction code


### Week 8:
- Goal:
    - Continue to explore transfer learning and improving accuracy
    - Improve real time prediction code
- Completed:
    - Real time prediction with trained model
    - Possibly need to adjust how frequently predictions are returned
    - Improved accuracy of pre-trained model
    - Transfer learned model has high accuracy on testing data from same session, but lower accuracy for testing data from different session
- Next steps:
    - Build physical hand for demonstrations
    - Continue to work on making network progressive (transfer learning)

### Week 9:
- Goal:
    - Work on transfer learning
- Completed:
    - Started on progressive neural networks
    - Experimented with intersession accuracy (recording data, then training and testing without moving armband vs. moving armband)
    - Made display of prediction more user friendly
- Next steps:
    - Try to finish PNN
    - Prepare video and portfolio

### Week 10:
- Goal:
    - Get PNN working
    - Get accurate real time prediction
    - Record video
- Completed:
    - Got video of real time prediction with fairly good accuracy
    - Did not complete PNN in time
