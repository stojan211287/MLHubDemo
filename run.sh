#!/bin/bash

# RUN DEPLOYMENT FLASK APP
python3 deployment.py &

# RUN MAIN APP
python3 app.py