from budapp.initializers.provider_seeder import ProviderSeeder
from budapp.initializers.user_seeder import UserSeeder


seeders = {
    "user": UserSeeder,
    "provider": ProviderSeeder,
}
