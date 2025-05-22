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
  token = ""
  cloud_id = "b1gp30rl3b9k0bo3qrbc"
  folder_id = "b1g6isgp470rh3ftc9vd"
}

resource "yandex_iam_service_account" "sa" {
  name = "terraform-test"
}

resource "yandex_resourcemanager_folder_iam_member" "sa-admin" {
  folder_id = "b1g6isgp470rh3ftc9vd"
  role = "storage.admin"
  member = "serviceAccount:${yandex_iam_service_account.sa.id}"
}

resource "yandex_iam_service_account_static_access_key" "sa-static-key" {
  service_account_id = yandex_iam_service_account.sa.id
  description = "static access key for object storage"
}

resource "yandex_storage_bucket" "test4" {
  access_key = yandex_iam_service_account_static_access_key.sa-static-key.access_key
  secret_key = yandex_iam_service_account_static_access_key.sa-static-key.secret_key
  bucket = "myuniquebucket12343336325354623443542654234214"
  tags = {
    test-1 = "1"
    test-2 = "lol"
  }
}