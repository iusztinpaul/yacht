# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: yacht/config/proto/train.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='yacht/config/proto/train.proto',
  package='yacht.config.proto',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1eyacht/config/proto/train.proto\x12\x12yacht.config.proto\"\x8e\x01\n\x0bTrainConfig\x12\r\n\x05steps\x18\x01 \x01(\x05\x12\x15\n\rlearning_rate\x18\x02 \x01(\x02\x12\x0f\n\x07n_steps\x18\x03 \x01(\x05\x12\x12\n\nbatch_size\x18\x04 \x01(\x05\x12\x10\n\x08n_epochs\x18\x05 \x01(\x05\x12\x10\n\x08log_freq\x18\x06 \x01(\x05\x12\x10\n\x08val_freq\x18\x07 \x01(\x05\x62\x06proto3'
)




_TRAINCONFIG = _descriptor.Descriptor(
  name='TrainConfig',
  full_name='yacht.config.proto.TrainConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='steps', full_name='yacht.config.proto.TrainConfig.steps', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='yacht.config.proto.TrainConfig.learning_rate', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='n_steps', full_name='yacht.config.proto.TrainConfig.n_steps', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='yacht.config.proto.TrainConfig.batch_size', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='n_epochs', full_name='yacht.config.proto.TrainConfig.n_epochs', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='log_freq', full_name='yacht.config.proto.TrainConfig.log_freq', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='val_freq', full_name='yacht.config.proto.TrainConfig.val_freq', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=55,
  serialized_end=197,
)

DESCRIPTOR.message_types_by_name['TrainConfig'] = _TRAINCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrainConfig = _reflection.GeneratedProtocolMessageType('TrainConfig', (_message.Message,), {
  'DESCRIPTOR' : _TRAINCONFIG,
  '__module__' : 'yacht.config.proto.train_pb2'
  # @@protoc_insertion_point(class_scope:yacht.config.proto.TrainConfig)
  })
_sym_db.RegisterMessage(TrainConfig)


# @@protoc_insertion_point(module_scope)
