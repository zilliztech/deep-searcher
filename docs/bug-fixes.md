# Bug Fix Report

## Overview
This document summarizes bugs found through the test suite and the fixes applied.

## Bugs Identified

1. **JiekouAI embedding default model mismatch**
   - **Symptom:** Unit tests expected `baai/bge-m3`, but runtime default used `qwen/qwen3-embedding-8b`.
   - **Impact:** Unexpected model selection and request payload mismatch for users relying on defaults.

2. **OpenAI client initialization leaked ambient base URL when explicit API key was provided**
   - **Symptom:** When `api_key` was passed directly, `OPENAI_BASE_URL` from environment was still applied.
   - **Impact:** Surprising and brittle behavior where explicit constructor usage could be silently redirected.

3. **XAI default model mismatch**
   - **Symptom:** Default in implementation was `grok-4`, while package behavior/tests expect `grok-2-latest`.
   - **Impact:** Behavior drift and compatibility issues for users relying on documented/default model.

## Fixes Applied

1. Updated `JiekouAIEmbedding` default model to `baai/bge-m3` and aligned `model_name` override fallback check.
2. Updated `OpenAI` initialization precedence so that when `api_key` is explicitly provided and `base_url` is not, `base_url` defaults to `None` rather than inheriting `OPENAI_BASE_URL`.
3. Updated `XAI` default model to `grok-2-latest`.

## Validation

- Ran targeted failing test modules and confirmed all pass.
- Ran complete test suite and confirmed all tests pass.

