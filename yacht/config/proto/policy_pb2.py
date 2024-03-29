# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: yacht/config/proto/policy.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from yacht.config.proto import feature_extractor_pb2 as yacht_dot_config_dot_proto_dot_feature__extractor__pb2
from yacht.config.proto import net_architecture_pb2 as yacht_dot_config_dot_proto_dot_net__architecture__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='yacht/config/proto/policy.proto',
  package='yacht.config.proto',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x1fyacht/config/proto/policy.proto\x12\x12yacht.config.proto\x1a*yacht/config/proto/feature_extractor.proto\x1a)yacht/config/proto/net_architecture.proto\"\xb7\x01\n\x0cPolicyConfig\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x15\n\ractivation_fn\x18\x02 \x01(\t\x12\x45\n\x11\x66\x65\x61ture_extractor\x18\x03 \x01(\x0b\x32*.yacht.config.proto.FeatureExtractorConfig\x12;\n\x08net_arch\x18\x04 \x01(\x0b\x32).yacht.config.proto.NetArchitectureConfigb\x06proto3'
  ,
  dependencies=[yacht_dot_config_dot_proto_dot_feature__extractor__pb2.DESCRIPTOR,yacht_dot_config_dot_proto_dot_net__architecture__pb2.DESCRIPTOR,])




_POLICYCONFIG = _descriptor.Descriptor(
  name='PolicyConfig',
  full_name='yacht.config.proto.PolicyConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='yacht.config.proto.PolicyConfig.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activation_fn', full_name='yacht.config.proto.PolicyConfig.activation_fn', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature_extractor', full_name='yacht.config.proto.PolicyConfig.feature_extractor', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='net_arch', full_name='yacht.config.proto.PolicyConfig.net_arch', index=3,
      number=4, type=11, cpp_type=10, label=1,
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
  serialized_start=143,
  serialized_end=326,
)

_POLICYCONFIG.fields_by_name['feature_extractor'].message_type = yacht_dot_config_dot_proto_dot_feature__extractor__pb2._FEATUREEXTRACTORCONFIG
_POLICYCONFIG.fields_by_name['net_arch'].message_type = yacht_dot_config_dot_proto_dot_net__architecture__pb2._NETARCHITECTURECONFIG
DESCRIPTOR.message_types_by_name['PolicyConfig'] = _POLICYCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PolicyConfig = _reflection.GeneratedProtocolMessageType('PolicyConfig', (_message.Message,), {
  'DESCRIPTOR' : _POLICYCONFIG,
  '__module__' : 'yacht.config.proto.policy_pb2'
  # @@protoc_insertion_point(class_scope:yacht.config.proto.PolicyConfig)
  })
_sym_db.RegisterMessage(PolicyConfig)


# @@protoc_insertion_point(module_scope)
