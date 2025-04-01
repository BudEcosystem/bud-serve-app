from budapp.initializers.base_keycloak_seeder import BaseKeycloakSeeder
from budapp.initializers.cloud_model_seeder import CloudModelSeeder
from budapp.initializers.cloud_provider_seeder import CloudProviderSeeder
from budapp.initializers.dataset_seeder import DatasetSeeder
from budapp.initializers.icon_seeder import IconSeeder
from budapp.initializers.provider_seeder import ProviderSeeder
from budapp.initializers.quantization_method_seeder import QuantizationMethodSeeder
from budapp.initializers.template_seeder import TemplateSeeder


seeders = {
    "keycloak": BaseKeycloakSeeder,
    # "user": UserSeeder,
    "provider": ProviderSeeder,
    "cloud_model": CloudModelSeeder,
    "icon": IconSeeder,
    "template": TemplateSeeder,
    "cloud_provider": CloudProviderSeeder,
    "quantization_method": QuantizationMethodSeeder,
    "datasets": DatasetSeeder
}
