"""
Django settings for quiz_gen_django project.

Generated by 'django-admin startproject' using Django 5.1.1.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.1/ref/settings/
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# Загружаем переменные из .env файла
load_dotenv()

# получаем SECRET_KEY
SECRET_KEY = os.getenv("SECRET_KEY")

MY_TRUSTED_CSRF = os.getenv("CSRF_TRUSTED_ORIGIN")
if MY_TRUSTED_CSRF is None:
    RuntimeError(f"Not all environment variables: missing CSRF_TRUSTED_ORIGIN value is '{MY_TRUSTED_CSRF}'")

INFERENCE_SERVER_MODE = os.getenv("INFERENCE_SERVER_MODE")
print(f"[INFO] INFERENCE_SERVER_MODE: {INFERENCE_SERVER_MODE}")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["*"]
CSRF_TRUSTED_ORIGINS = [
    MY_TRUSTED_CSRF,
    'https://smartquizgenerator.onrender.com'
]


# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "quiz_gen_django",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "quiz_gen_django.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": ["templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "quiz_gen_django.wsgi.application"
# ASGI_APPLICATION = "quiz_gen_django.asgi.application"

# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

RUNNING_ON_VERCEL = os.getenv("RUNNING_ON_VERCEL", 'False').lower() in ('true', '1', 't')
print(f"[info] RUNNING_ON_VERCEL: {RUNNING_ON_VERCEL}")
DATABASES = {
    "default": {
        # [ATTENTION]: see vercel SQLite issue:
        #   - https://github.com/vercel/vercel/issues/2860#issuecomment-522087251
        #   - https://vercel.com/guides/is-sqlite-supported-in-vercel
        "ENGINE": "" if RUNNING_ON_VERCEL else "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = "static/"
STATICFILES_DIRS = [
    BASE_DIR / "static",
]

STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
