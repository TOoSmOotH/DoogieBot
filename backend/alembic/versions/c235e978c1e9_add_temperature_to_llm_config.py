"""add temperature to llm_config

Revision ID: c235e978c1e9
Revises: 202503281453
Create Date: 2025-04-01 03:34:50.767068

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = 'c235e978c1e9'
down_revision = '202503281453'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Removed incorrect drop operations
    op.add_column('llm_config', sa.Column('temperature', sa.Float(), nullable=True))
    # Removed incorrect alter_column and drop_index operations
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Removed incorrect create_index and alter_column operations
    op.drop_column('llm_config', 'temperature')
    # Removed incorrect create_table operations
    # ### end Alembic commands ###