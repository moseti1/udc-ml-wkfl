# serialize image function
import json
import boto3
import base64

s3 = boto3.client('s3')

BUCKET_NAME = 'sagemaker-workflow2' # bucket name


def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_input_uri']
    bucket = BUCKET_NAME
    # Download the data from s3 to /tmp/image.png
    file_name = '/tmp/image.png'
    
    s3.download_file(bucket,key, file_name)
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }
#EOF


# classify images function 
import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

ENDPOINT = "image-classification-2021-11-14-03-08-32-250" # name of endpoint to use

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])
    endpoint = ENDPOINT
    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(
    endpoint,
    sagemaker_session=sagemaker.Session(),
    )

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    event['inferences'] = inferences.decode('utf-8')
    
    return {
        'statusCode': 200,
        'body': {
            "inferences": event['inferences']
            
        }
    }
#EOF

# check confidence threshold
import json

THRESHOLD = 0.97


def lambda_handler(event, context):
    meets_threshold = None
    # Grab the inferences from the event
    inferences = json.loads(event['inferences'])
    # Check if any values in our inferences are above THRESHOLD
    for ifer in inferences:
        if infer > THRESHOLD:
            meets_threshold = True
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

#EOF

