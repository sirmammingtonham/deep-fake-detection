from bs4 import BeautifulSoup
import cv2
import numpy as np
import urllib
import re
import requests
import json
import os

def get_elements(page_link):
	# implement try catch

	page_response = requests.get(page_link, timeout=5)
	
	page_content = BeautifulSoup(page_response.content, "html.parser")
	
	paragraphs = page_content.find_all("p")
	clean_paragraphs = []

	if page_content.find('figure') != None:
		images = page_content.find('figure').find_all('img', src=True)
		figure = True
	else:
		images = page_content.find_all('img', src=True)
		figure = False

	clean_images = []

	for pg in paragraphs:
		p = pg.text
		o = re.sub(r"\s+", " ", p)
		clean_paragraphs.append(o)
	print (len(clean_paragraphs), "paragraphs found.")

	for im in images:
		viable = True

		url = im['src']
		url = url.strip('//')
		if 'http://' not in url and 'https://' not in url:
			url = 'http://' + url
		url.replace('&amp;', '&')

		print(url)

		if not figure and not url.endswith(('.jpg', '.png', '.jpeg', '.gif')):
			viable = False 

		if viable:
			opener = urllib.request.build_opener()
			opener.addheaders = [('User-agent', 'Mozilla/5.0')]
			urllib.request.install_opener(opener)
			urllib.request.urlretrieve(url, "temp.jpg")

			# Get user supplied values
			imagePath = "temp.jpg"
			cascPath = "/home/ethanj217/deep-fake-detection/app/haarcascade_frontalface_default.xml"

			# Create the haar cascade
			faceCascade = cv2.CascadeClassifier(cascPath)

			# Read the image
			image = cv2.imread(imagePath)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# Detect faces in the image
			faces = faceCascade.detectMultiScale(
			    gray,
			    scaleFactor=1.1,
			    minNeighbors=5,
			    minSize=(30, 30)
			)

			print("Found {0} faces!".format(len(faces)))

			if len(faces) > 0:
				clean_images.append(url)
		else:
			print("not viable")

	print (len(clean_images), "images found.")

	if os.path.exists("temp.jpg"):
  		os.remove("temp.jpg")

	return [clean_paragraphs, clean_images, None]
