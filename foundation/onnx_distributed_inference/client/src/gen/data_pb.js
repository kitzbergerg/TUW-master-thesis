// @generated by protoc-gen-es v2.4.0
// @generated from file data.proto (package websocket, syntax proto3)
/* eslint-disable */

import { fileDesc, messageDesc } from "@bufbuild/protobuf/codegenv1";

/**
 * Describes the file data.proto.
 */
export const file_data = /*@__PURE__*/
  fileDesc("CgpkYXRhLnByb3RvEgl3ZWJzb2NrZXQijAMKEFdlYlNvY2tldE1lc3NhZ2USNAoPY29ubmVjdGVkX3VzZXJzGAEgASgLMhkud2Vic29ja2V0LkNvbm5lY3RlZFVzZXJzSAASKwoKaW5pdGlhbGl6ZRgCIAEoCzIVLndlYnNvY2tldC5Jbml0aWFsaXplSAASNAoPaW5pdGlhbGl6ZV9kb25lGAMgASgLMhkud2Vic29ja2V0LkluaXRpYWxpemVEb25lSAASOAoRaW5mZXJlbmNlX3JlcXVlc3QYBCABKAsyGy53ZWJzb2NrZXQuSW5mZXJlbmNlUmVxdWVzdEgAEi0KC2NvbXB1dGF0aW9uGAUgASgLMhYud2Vic29ja2V0LkNvbXB1dGF0aW9uSAASNgoQaW5mZXJlbmNlX3Jlc3VsdBgGIAEoCzIaLndlYnNvY2tldC5JbmZlcmVuY2VSZXN1bHRIABI2ChBpbnZhbGlkYXRlX2NhY2hlGAcgASgLMhoud2Vic29ja2V0LkludmFsaWRhdGVDYWNoZUgAQgYKBGtpbmQiIQoOQ29ubmVjdGVkVXNlcnMSDwoHbWVzc2FnZRgBIAEoDSJGCgpJbml0aWFsaXplEg8KB25vZGVfaWQYASABKAkSJwoHbWVzc2FnZRgCIAEoCzIWLndlYnNvY2tldC5Nb2RlbENvbmZpZyIhCg5Jbml0aWFsaXplRG9uZRIPCgdub2RlX2lkGAEgASgJIiMKEEluZmVyZW5jZVJlcXVlc3QSDwoHbWVzc2FnZRgBIAEoCSI9CgtDb21wdXRhdGlvbhIuCgdtZXNzYWdlGAEgASgLMh0ud2Vic29ja2V0LkNvbXB1dGF0aW9uTWVzc2FnZSIiCg9JbmZlcmVuY2VSZXN1bHQSDwoHbWVzc2FnZRgBIAEoCSI5Cg9JbnZhbGlkYXRlQ2FjaGUSFwoKcmVxdWVzdF9pZBgBIAEoCUgAiAEBQg0KC19yZXF1ZXN0X2lkIlUKC01vZGVsQ29uZmlnEhEKCW1vZGVsX3VyaRgBIAEoCRIzCg1leHRlcm5hbF9kYXRhGAIgAygLMhwud2Vic29ja2V0LkV4dGVybmFsRGF0YUVudHJ5IqgBChJDb21wdXRhdGlvbk1lc3NhZ2USDwoHbm9kZV9pZBgBIAEoCRISCgpyZXF1ZXN0X2lkGAIgASgJEisKBWZpcnN0GAMgASgLMhoud2Vic29ja2V0LkZpcnN0TW9kZWxJbnB1dEgAEjgKDGludGVybWVkaWF0ZRgEIAEoCzIgLndlYnNvY2tldC5JbnRlcm1lZGlhdGVNb2RlbERhdGFIAEIGCgRkYXRhIi8KEUV4dGVybmFsRGF0YUVudHJ5EgwKBHBhdGgYASABKAkSDAoEZGF0YRgCIAEoCSIkCg9GaXJzdE1vZGVsSW5wdXQSEQoJaW5wdXRfaWRzGAEgAygNIpoBChVJbnRlcm1lZGlhdGVNb2RlbERhdGESNgoDbWFwGAEgAygLMikud2Vic29ja2V0LkludGVybWVkaWF0ZU1vZGVsRGF0YS5NYXBFbnRyeRpJCghNYXBFbnRyeRILCgNrZXkYASABKAkSLAoFdmFsdWUYAiABKAsyHS53ZWJzb2NrZXQuSW50ZXJtZWRpYXRlUmVzdWx0OgI4ASIwChJJbnRlcm1lZGlhdGVSZXN1bHQSDAoEZGF0YRgBIAMoAhIMCgRkaW1zGAIgAygNYgZwcm90bzM");

/**
 * Describes the message websocket.WebSocketMessage.
 * Use `create(WebSocketMessageSchema)` to create a new message.
 */
export const WebSocketMessageSchema = /*@__PURE__*/
  messageDesc(file_data, 0);

/**
 * Describes the message websocket.ConnectedUsers.
 * Use `create(ConnectedUsersSchema)` to create a new message.
 */
export const ConnectedUsersSchema = /*@__PURE__*/
  messageDesc(file_data, 1);

/**
 * Describes the message websocket.Initialize.
 * Use `create(InitializeSchema)` to create a new message.
 */
export const InitializeSchema = /*@__PURE__*/
  messageDesc(file_data, 2);

/**
 * Describes the message websocket.InitializeDone.
 * Use `create(InitializeDoneSchema)` to create a new message.
 */
export const InitializeDoneSchema = /*@__PURE__*/
  messageDesc(file_data, 3);

/**
 * Describes the message websocket.InferenceRequest.
 * Use `create(InferenceRequestSchema)` to create a new message.
 */
export const InferenceRequestSchema = /*@__PURE__*/
  messageDesc(file_data, 4);

/**
 * Describes the message websocket.Computation.
 * Use `create(ComputationSchema)` to create a new message.
 */
export const ComputationSchema = /*@__PURE__*/
  messageDesc(file_data, 5);

/**
 * Describes the message websocket.InferenceResult.
 * Use `create(InferenceResultSchema)` to create a new message.
 */
export const InferenceResultSchema = /*@__PURE__*/
  messageDesc(file_data, 6);

/**
 * Describes the message websocket.InvalidateCache.
 * Use `create(InvalidateCacheSchema)` to create a new message.
 */
export const InvalidateCacheSchema = /*@__PURE__*/
  messageDesc(file_data, 7);

/**
 * Describes the message websocket.ModelConfig.
 * Use `create(ModelConfigSchema)` to create a new message.
 */
export const ModelConfigSchema = /*@__PURE__*/
  messageDesc(file_data, 8);

/**
 * Describes the message websocket.ComputationMessage.
 * Use `create(ComputationMessageSchema)` to create a new message.
 */
export const ComputationMessageSchema = /*@__PURE__*/
  messageDesc(file_data, 9);

/**
 * Describes the message websocket.ExternalDataEntry.
 * Use `create(ExternalDataEntrySchema)` to create a new message.
 */
export const ExternalDataEntrySchema = /*@__PURE__*/
  messageDesc(file_data, 10);

/**
 * Describes the message websocket.FirstModelInput.
 * Use `create(FirstModelInputSchema)` to create a new message.
 */
export const FirstModelInputSchema = /*@__PURE__*/
  messageDesc(file_data, 11);

/**
 * Describes the message websocket.IntermediateModelData.
 * Use `create(IntermediateModelDataSchema)` to create a new message.
 */
export const IntermediateModelDataSchema = /*@__PURE__*/
  messageDesc(file_data, 12);

/**
 * Describes the message websocket.IntermediateResult.
 * Use `create(IntermediateResultSchema)` to create a new message.
 */
export const IntermediateResultSchema = /*@__PURE__*/
  messageDesc(file_data, 13);

