# lung_nodule_segmentation_and_classification
environments： 01~04 .ipynb - Colab<br/>
               interface - local environment<br/>
               <br/>
LUNA-16 dataset is uploaded to kaggle<br/>
<br/>
you can choose how many subsets you'd like to use in 01.ipynb, and directly download the data from kaggle to colab<br/>
You should run the 01 to 04.ipynb files in order, which requires creating some paths in Google Drive or changing them to your paths<br/>
The user interface requires running locally. Please create a virtual environment containing tensorflow and pytorch, and make sure to save the correct model and path.<br/>
<br/>
we combined labels from LUNA16 and LUNA22, they are the subset of LIDC-IDRI<br/>
The processed labels are stored in csv files<br/>
<br/>
Make sure you read the instructions above before running each piece of code<br/>
thanks to https://github.com/abhikrm0102/Lung-Nodules-Detection-and-Classification-using-UNet-DenseNet.git this project help us a lot<br/>
