import logging as log
import paho.mqtt.client as mqtt
import socket


class Mqtt:
    def __init__(self):
        self.HOSTNAME = socket.gethostname()
        self.IPADDRESS = socket.gethostbyname(self.HOSTNAME)
        self.MQTT_HOST = self.IPADDRESS
        self.MQTT_PORT = 1884
        self.MQTT_KEEPALIVE_INTERVAL = 60
        self.client = mqtt.Client()

    def connect(self):
        log.info("Connected to MQTT server")
        self.client.connect(self.MQTT_HOST, self.MQTT_PORT, self.MQTT_KEEPALIVE_INTERVAL)
        return self.client

    def disconnect(self):
        log.info("DisConnected from MQTT server")
        return self.client.disconnect()

    def publishtomqtt(self, topic, content):
        log.info("Published content to MQTT topic", topic)
        return self.client.publish(topic, content)
