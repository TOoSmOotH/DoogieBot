"""add reranked_top_n to llm_config

Revision ID: a1b2c3d4e5f6
Revises: c235e978c1e9
Create Date: 2025-04-01 15:54:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'c235e978c1e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('llm_config', schema=None) as batch_op:
        batch_op.add_column(sa.Column('reranked_top_n', sa.Integer(), nullable=True))

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('llm_config', schema=None) as batch_op:
        batch_op.drop_column('reranked_top_n')

    # ### end Alembic commands ###