#  -----------------------------------------------------------------------------
#  Copyright (c) 2024 Bud Ecosystem Inc.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  -----------------------------------------------------------------------------

"""The playground ops package, containing essential business logic, services, and routing configurations for the playground ops."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, Uuid, ForeignKey, Integer, Float, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from budapp.commons.database import Base, TimestampMixin


class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    chat_setting_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("chat_settings.id", ondelete="CASCADE"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    modified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    chat_setting: Mapped["ChatSetting"] = relationship(
        "ChatSetting", back_populates="chat_sessions", foreign_keys=[chat_setting_id]
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="chat_session", cascade="all, delete-orphan"
    )
    notes: Mapped[list["Note"]] = relationship("Note", back_populates="chat_session", cascade="all, delete-orphan")


class Note(Base, TimestampMixin):
    __tablename__ = "notes"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    chat_session_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    note: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    modified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    chat_session = relationship("ChatSession", back_populates="notes")


class Message(Base, TimestampMixin):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    chat_session_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )

    prompt: Mapped[str] = mapped_column(String, nullable=False)
    response: Mapped[list[dict]] = mapped_column(JSON, nullable=False)

    deployment_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    parent_message_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("messages.id", ondelete="CASCADE"),  # Ensures all children are deleted
        nullable=True,
    )

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    token_per_sec: Mapped[float] = mapped_column(Float, nullable=True)
    ttft: Mapped[float] = mapped_column(Float, nullable=True)
    tpot: Mapped[float] = mapped_column(Float, nullable=True)
    e2e_latency: Mapped[float] = mapped_column(Float, nullable=True)
    is_cache: Mapped[bool] = mapped_column(Boolean, nullable=False)
    harmfullness: Mapped[float] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float] = mapped_column(Float, nullable=True)

    # upvotes: Mapped[int] = mapped_column(Integer, nullable=True)
    # downvotes: Mapped[int] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    modified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    chat_session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")
    parent_message: Mapped["Message"] = relationship(
        "Message", remote_side=[id], backref=backref("child_messages", cascade="all, delete")
    )


class ChatSetting(Base, TimestampMixin):
    __tablename__ = "chat_settings"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    preset_name: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    modified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationship back to chat sessions using these settings.
    chat_sessions: Mapped[list["ChatSession"]] = relationship("ChatSession", back_populates="chat_setting")
