/*++
Copyright (c) Microsoft Corporation

Module Name:
- GlyphWidth.hpp

Abstract:
- Helpers for determining the width of a particular string of chars.

*/

bool IsGlyphFullWidth(const std::wstring_view glyph);
bool IsGlyphFullWidth(const wchar_t wch);
void SetGlyphWidthFallback(std::function<bool(std::wstring_view)> pfnFallback);
