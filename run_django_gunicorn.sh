#!/bin/bash

gunicorn --bind 0.0.0.0:10000 --timeout 450 quiz_gen_django.wsgi:application
