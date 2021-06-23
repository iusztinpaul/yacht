# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: yacht/config/proto/config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from yacht.config.proto import input_pb2 as yacht_dot_config_dot_proto_dot_input__pb2
from yacht.config.proto import environment_pb2 as yacht_dot_config_dot_proto_dot_environment__pb2
from yacht.config.proto import agent_pb2 as yacht_dot_config_dot_proto_dot_agent__pb2
from yacht.config.proto import train_pb2 as yacht_dot_config_dot_proto_dot_train__pb2
from yacht.config.proto import meta_pb2 as yacht_dot_config_dot_proto_dot_meta__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='yacht/config/proto/config.proto',
  package='yacht.config.proto',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1fyacht/config/proto/config.proto\x12\x12yacht.config.proto\x1a\x1eyacht/config/proto/input.proto\x1a$yacht/config/proto/environment.proto\x1a\x1eyacht/config/proto/agent.proto\x1a\x1eyacht/config/proto/train.proto\x1a\x1dyacht/config/proto/meta.proto\"\x82\x02\n\x06\x43onfig\x12.\n\x05input\x18\x01 \x01(\x0b\x32\x1f.yacht.config.proto.InputConfig\x12:\n\x0b\x65nvironment\x18\x02 \x01(\x0b\x32%.yacht.config.proto.EnvironmentConfig\x12.\n\x05\x61gent\x18\x03 \x01(\x0b\x32\x1f.yacht.config.proto.AgentConfig\x12.\n\x05train\x18\x04 \x01(\x0b\x32\x1f.yacht.config.proto.TrainConfig\x12,\n\x04meta\x18\x05 \x01(\x0b\x32\x1e.yacht.config.proto.MetaConfigb\x06proto3')
  ,
  dependencies=[yacht_dot_config_dot_proto_dot_input__pb2.DESCRIPTOR,yacht_dot_config_dot_proto_dot_environment__pb2.DESCRIPTOR,yacht_dot_config_dot_proto_dot_agent__pb2.DESCRIPTOR,yacht_dot_config_dot_proto_dot_train__pb2.DESCRIPTOR,yacht_dot_config_dot_proto_dot_meta__pb2.DESCRIPTOR,])




_CONFIG = _descriptor.Descriptor(
  name='Config',
  full_name='yacht.config.proto.Config',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='input', full_name='yacht.config.proto.Config.input', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='environment', full_name='yacht.config.proto.Config.environment', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='agent', full_name='yacht.config.proto.Config.agent', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train', full_name='yacht.config.proto.Config.train', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='meta', full_name='yacht.config.proto.Config.meta', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=221,
  serialized_end=479,
)

_CONFIG.fields_by_name['input'].message_type = yacht_dot_config_dot_proto_dot_input__pb2._INPUTCONFIG
_CONFIG.fields_by_name['environment'].message_type = yacht_dot_config_dot_proto_dot_environment__pb2._ENVIRONMENTCONFIG
_CONFIG.fields_by_name['agent'].message_type = yacht_dot_config_dot_proto_dot_agent__pb2._AGENTCONFIG
_CONFIG.fields_by_name['train'].message_type = yacht_dot_config_dot_proto_dot_train__pb2._TRAINCONFIG
_CONFIG.fields_by_name['meta'].message_type = yacht_dot_config_dot_proto_dot_meta__pb2._METACONFIG
DESCRIPTOR.message_types_by_name['Config'] = _CONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Config = _reflection.GeneratedProtocolMessageType('Config', (_message.Message,), dict(
  DESCRIPTOR = _CONFIG,
  __module__ = 'yacht.config.proto.config_pb2'
  # @@protoc_insertion_point(class_scope:yacht.config.proto.Config)
  ))
_sym_db.RegisterMessage(Config)


# @@protoc_insertion_point(module_scope)
