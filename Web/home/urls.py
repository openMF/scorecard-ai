# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# AI-Assisted Development Notice:
# Portions of this file were developed with the assistance of GitHub Copilot.
# All AI-generated code was reviewed, tested, and validated by the contributor.

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/v1/predict', views.predict_api_v1, name='predict_api_v1'),
]
