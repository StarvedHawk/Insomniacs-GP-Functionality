import psutil
import json
import os
import pandas as pd
from pynput import keyboard
import schedule
import time
#run pip install for psutil, pandas, pynput, and schedule

def processLogger(): # This function saves all running into 3 json files

	print('Running processLogger')
	# Read whitelist and blacklist
	blacklistData = pd.read_csv('blacklist.csv', sep=',', engine='python', header=None)
	dfb = pd.DataFrame(blacklistData)
	blacklist = dfb.values.tolist()

	whitelistData = pd.read_csv('whitelist.csv', engine='python', header=None)
	dfw = pd.DataFrame(whitelistData)
	whitelist = dfw.values.tolist()

	# Find the the existing file and delete it
	if os.path.exists("whitelistedProcess.json"): 
		os.remove("whitelistedProcess.json")

	if os.path.exists("blacklistedProcess.json"):
		os.remove("blacklistedProcess.json")

	if os.path.exists("unlistedProcess.json"):
		os.remove("unlistedProcess.json")

	#The following code gets the current running process and saves them to different files depending on if they are present in the white or black list 
	# or if they arent
	for proc in psutil.process_iter():
		try:
			processName = proc.name()	
				
			if any(processName in word for word in blacklist):
				blacklisted = {"Process Name" : processName} 
				print('Blacklist detected')
				with open("blacklistedProcess.json","a") as file:
					json.dump(blacklisted, file)	
			
			elif any(processName in word for word in whitelist):
				whitelisted = {"Process Name" : processName}
				with open("whitelistedProcess.json","a") as file:
					json.dump(whitelisted, file)
		
			else:
				process = {"Process Name" : processName}
				with open("unlistedProcess.json","a") as file:
					json.dump(process, file)


		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
			pass

copyCounter = 0
pasteCounter = 0

def copyPasteLogger(): # The following code tracks how many times ctrl + c and ctrl + v are pressed
	
	if os.path.exists("copyLogged.json"): 
		os.remove("copyLogged.json")
	if os.path.exists("pasteLogged.json"): 
		os.remove("pasteLogged.json")
	
	# The key combination to check
	copyComby = {keyboard.KeyCode.from_char('c') , keyboard.Key.ctrl_l}
	pasteComby = {keyboard.KeyCode.from_char('v') , keyboard.Key.ctrl_l}
	# The currently active modifiers
	current = set()

	# This code is for handling what happens when you press a key
	def on_press(key):
		try:
			if key in copyComby:
				current.add(key)
				if all(k in current for k in copyComby):
					global copyCounter
					copyCounter += 1
					copyLogged = {"Pressed ctrl + c" : copyCounter}
					with open("copyLogged.json","w") as file:
						json.dump(copyLogged, file)

			if key in pasteComby:
				current.add(key)
				if all(k in current for k in pasteComby):
					global pasteCounter
					pasteCounter += 1
					pasteLogged = {"Pressed ctrl + v" : pasteCounter}
					with open("pasteLogged.json","w") as file:
						json.dump(pasteLogged, file)
		except KeyboardInterrupt:
			pass	

	def on_release(key):
		try:
			current.remove(key)
		except (KeyError,KeyboardInterrupt):
			pass
	
	def win32_event_filter(msg, data):
		'''print(msg, data)'''
	with keyboard.Listener(on_press=on_press, on_release=on_release, win32_event_filter=win32_event_filter, suppress=False) as listener:
		listener.join()


processLogger()
copyPasteLogger()
schedule.every(45).seconds.do(processLogger)	#schedules code to run

while True:
	schedule.run_pending()
	time.sleep(1)

