import os
from logging.config import fileConfig

from alembic import context
from pydantic import PostgresDsn
from sqlalchemy import create_engine, engine_from_config, pool
from sqlalchemy_utils import create_database, database_exists

from budapp.commons import Base


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_psql_url() -> PostgresDsn:
    if os.getenv("PSQL_HOST") is None or os.getenv("PSQL_PORT") is None or os.getenv("PSQL_DB_NAME") is None:
        raise ValueError("PSQL_HOST, PSQL_PORT, and PSQL_DB_NAME must be set")

    return PostgresDsn.build(
        scheme="postgresql+psycopg",
        username=os.getenv("SECRETS_PSQL_USER"),
        password=os.getenv("SECRETS_PSQL_PASSWORD"),
        host=os.getenv("PSQL_HOST"),
        port=int(os.getenv("PSQL_PORT")),
        path=os.getenv("PSQL_DB_NAME"),
    ).__str__()


def create_db() -> None:
    """Create the database."""
    DATABASE_URL = get_psql_url()

    engine = create_engine(DATABASE_URL)

    # Check if the database exists, and create it if not
    if not database_exists(engine.url):
        create_database(engine.url)
        print(f"Database created at {DATABASE_URL}")
    else:
        print(f"Database already exists at {DATABASE_URL}")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_psql_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_psql_url()

    print(configuration["sqlalchemy.url"])

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


create_db()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
