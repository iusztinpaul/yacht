# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: yacht/config/proto/input.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from yacht.config.proto import backtest_pb2 as yacht_dot_config_dot_proto_dot_backtest__pb2
from yacht.config.proto import period_pb2 as yacht_dot_config_dot_proto_dot_period__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='yacht/config/proto/input.proto',
  package='yacht.config.proto',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x1eyacht/config/proto/input.proto\x12\x12yacht.config.proto\x1a!yacht/config/proto/backtest.proto\x1a\x1fyacht/config/proto/period.proto\"\xf5\x05\n\x0bInputConfig\x12\x0e\n\x06market\x18\x01 \x01(\t\x12\x15\n\rmarket_mixins\x18\x02 \x03(\t\x12\x0f\n\x07\x64\x61taset\x18\x03 \x01(\t\x12\x16\n\x0eis_multi_asset\x18\x04 \x01(\x08\x12\x1e\n\x16num_assets_per_dataset\x18\x05 \x01(\x05\x12\x0e\n\x06scaler\x18\x06 \x01(\t\x12\x19\n\x11scale_on_interval\x18\x07 \x01(\t\x12\x0f\n\x07tickers\x18\x08 \x03(\t\x12\x19\n\x11\x66ine_tune_tickers\x18\t \x03(\t\x12\x11\n\tintervals\x18\n \x03(\t\x12\x10\n\x08\x66\x65\x61tures\x18\x0b \x03(\t\x12\x1e\n\x16\x64\x65\x63ision_price_feature\x18\x0c \x01(\t\x12\x16\n\x0etake_action_at\x18\r \x01(\t\x12\x1c\n\x14technical_indicators\x18\x0e \x03(\t\x12\r\n\x05start\x18\x0f \x01(\t\x12\x0b\n\x03\x65nd\x18\x10 \x01(\t\x12\x15\n\rperiod_length\x18\x11 \x01(\t\x12\x13\n\x0bwindow_size\x18\x12 \x01(\x05\x12\x19\n\x11window_transforms\x18\x13 \x03(\t\x12\x13\n\x0bnum_periods\x18\x14 \x01(\x05\x12\x38\n\x0erender_periods\x18\x15 \x03(\x0b\x32 .yacht.config.proto.PeriodConfig\x12\x16\n\x0erender_tickers\x18\x16 \x03(\t\x12\x18\n\x10include_weekends\x18\x17 \x01(\x08\x12\x1e\n\x16validation_split_ratio\x18\x18 \x01(\x01\x12\x1c\n\x14\x62\x61\x63ktest_split_ratio\x18\x19 \x01(\x01\x12\x15\n\rembargo_ratio\x18\x1a \x01(\x01\x12\x1b\n\x13train_on_validation\x18\x1b \x01(\x08\x12\x34\n\x08\x62\x61\x63ktest\x18\x1c \x01(\x0b\x32\".yacht.config.proto.BacktestConfig\x12\x18\n\x10\x61ttached_tickers\x18\x1d \x03(\tb\x06proto3'
  ,
  dependencies=[yacht_dot_config_dot_proto_dot_backtest__pb2.DESCRIPTOR,yacht_dot_config_dot_proto_dot_period__pb2.DESCRIPTOR,])




_INPUTCONFIG = _descriptor.Descriptor(
  name='InputConfig',
  full_name='yacht.config.proto.InputConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='market', full_name='yacht.config.proto.InputConfig.market', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='market_mixins', full_name='yacht.config.proto.InputConfig.market_mixins', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataset', full_name='yacht.config.proto.InputConfig.dataset', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_multi_asset', full_name='yacht.config.proto.InputConfig.is_multi_asset', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_assets_per_dataset', full_name='yacht.config.proto.InputConfig.num_assets_per_dataset', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scaler', full_name='yacht.config.proto.InputConfig.scaler', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale_on_interval', full_name='yacht.config.proto.InputConfig.scale_on_interval', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tickers', full_name='yacht.config.proto.InputConfig.tickers', index=7,
      number=8, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fine_tune_tickers', full_name='yacht.config.proto.InputConfig.fine_tune_tickers', index=8,
      number=9, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='intervals', full_name='yacht.config.proto.InputConfig.intervals', index=9,
      number=10, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='features', full_name='yacht.config.proto.InputConfig.features', index=10,
      number=11, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='decision_price_feature', full_name='yacht.config.proto.InputConfig.decision_price_feature', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='take_action_at', full_name='yacht.config.proto.InputConfig.take_action_at', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='technical_indicators', full_name='yacht.config.proto.InputConfig.technical_indicators', index=13,
      number=14, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start', full_name='yacht.config.proto.InputConfig.start', index=14,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end', full_name='yacht.config.proto.InputConfig.end', index=15,
      number=16, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='period_length', full_name='yacht.config.proto.InputConfig.period_length', index=16,
      number=17, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='window_size', full_name='yacht.config.proto.InputConfig.window_size', index=17,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='window_transforms', full_name='yacht.config.proto.InputConfig.window_transforms', index=18,
      number=19, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_periods', full_name='yacht.config.proto.InputConfig.num_periods', index=19,
      number=20, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='render_periods', full_name='yacht.config.proto.InputConfig.render_periods', index=20,
      number=21, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='render_tickers', full_name='yacht.config.proto.InputConfig.render_tickers', index=21,
      number=22, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='include_weekends', full_name='yacht.config.proto.InputConfig.include_weekends', index=22,
      number=23, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='validation_split_ratio', full_name='yacht.config.proto.InputConfig.validation_split_ratio', index=23,
      number=24, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='backtest_split_ratio', full_name='yacht.config.proto.InputConfig.backtest_split_ratio', index=24,
      number=25, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='embargo_ratio', full_name='yacht.config.proto.InputConfig.embargo_ratio', index=25,
      number=26, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_on_validation', full_name='yacht.config.proto.InputConfig.train_on_validation', index=26,
      number=27, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='backtest', full_name='yacht.config.proto.InputConfig.backtest', index=27,
      number=28, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attached_tickers', full_name='yacht.config.proto.InputConfig.attached_tickers', index=28,
      number=29, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=123,
  serialized_end=880,
)

_INPUTCONFIG.fields_by_name['render_periods'].message_type = yacht_dot_config_dot_proto_dot_period__pb2._PERIODCONFIG
_INPUTCONFIG.fields_by_name['backtest'].message_type = yacht_dot_config_dot_proto_dot_backtest__pb2._BACKTESTCONFIG
DESCRIPTOR.message_types_by_name['InputConfig'] = _INPUTCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InputConfig = _reflection.GeneratedProtocolMessageType('InputConfig', (_message.Message,), {
  'DESCRIPTOR' : _INPUTCONFIG,
  '__module__' : 'yacht.config.proto.input_pb2'
  # @@protoc_insertion_point(class_scope:yacht.config.proto.InputConfig)
  })
_sym_db.RegisterMessage(InputConfig)


# @@protoc_insertion_point(module_scope)
