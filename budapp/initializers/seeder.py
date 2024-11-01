from budapp.initializers.cloud_model_seeder import CloudModelSeeder
from budapp.initializers.icon_seeder import IconSeeder
from budapp.initializers.provider_seeder import ProviderSeeder
from budapp.initializers.user_seeder import UserSeeder


seeders = {
    "user": UserSeeder,
    "provider": ProviderSeeder,
    "cloud_model": CloudModelSeeder,
    "icon": IconSeeder,
}
