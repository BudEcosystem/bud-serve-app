"""Updated model table.

Revision ID: b2a743972235
Revises: 55602d4240bb
Create Date: 2024-11-20 08:10:03.255087

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b2a743972235"
down_revision: Union[str, None] = "55602d4240bb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("cloud_model", "icon")
    op.add_column("model", sa.Column("provider_id", sa.Uuid(), nullable=True))
    op.alter_column("model", "icon", existing_type=sa.VARCHAR(), nullable=True)
    op.create_foreign_key(op.f("fk_model_provider_id_provider"), "model", "provider", ["provider_id"], ["id"])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(op.f("fk_model_provider_id_provider"), "model", type_="foreignkey")
    op.alter_column("model", "icon", existing_type=sa.VARCHAR(), nullable=False)
    op.drop_column("model", "provider_id")
    op.add_column("cloud_model", sa.Column("icon", sa.VARCHAR(), autoincrement=False, nullable=False))
    # ### end Alembic commands ###
