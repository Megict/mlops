terraform {
    required_providers {
        yandex = {
            source = "yandex-cloud/yandex"
            }
        }
    required_version = ">= 0.13"
}

provider "yandex" {
zone = "ru-central1-d"
token = "y0__xD64LVXGMHdEyDhvMbzEo7ZyKdCi2nwTP7Y04fEHMIDdhYb"
cloud_id = "b1gc7902ruhm9dkat3hc"
folder_id = "b1ge6rtd01ko22jheuqp"
}

resource "yandex_iam_service_account" "sa" {
name = "terraform-test"
}

resource "yandex_resourcemanager_folder_iam_member" "sa-admin" {
folder_id = "b1ge6rtd01ko22jheuqp"
role = "storage.admin"
member = "serviceAccount:${yandex_iam_service_account.sa.id}"
}

resource "yandex_iam_service_account_static_access_key" "sa-static-key" {
service_account_id = yandex_iam_service_account.sa.id
description = "static access key for object storage"
}

resource "yandex_storage_bucket" "test" {
access_key = yandex_iam_service_account_static_access_key.sa-static-key.access_key
secret_key = yandex_iam_service_account_static_access_key.sa-static-key.secret_key
bucket = "my-unique-bucket-2025"
max_size = 0
default_storage_class = "standard"
tags = {
test-1 = "1"
}
}