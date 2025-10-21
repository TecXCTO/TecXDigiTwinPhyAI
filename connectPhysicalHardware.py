import boto3
iot = boto3.client('iot-data', region_name='us-east-1')
iot.publish(
    topic="robot/command",
    qos=1,
    payload='{"action": "move_forward"}'
)
