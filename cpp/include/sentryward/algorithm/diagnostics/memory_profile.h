#pragma once
#ifdef SENTRYWARD_DEVELOPMENT_DIAGNOSTICS
#include "sentryward/development/memory_profile.h"
#else
#include <cstddef>
namespace sentryward::algorithm {
class SharedString; struct SourceArtifact; class LoweredProgram; class ConstantAddressModel; class ReferenceTrace;
inline bool MemoryProfileEnabled() { return false; }
inline void MemoryCheckpoint(const char*,const char* = "") {}
struct MemoryReport {
 template<class... T> explicit MemoryReport(T&&...) {}
 template<class... T> void Add(T&&...) {}
 template<class... T> void Object(T&&...) {}
 template<class... T> void Vector(T&&...) {}
 template<class... T> void Tree(T&&...) {}
 template<class... T> void Hash(T&&...) {}
 template<class... T> void List(T&&...) {}
 template<class... T> void String(T&&...) {}
 template<class... T> void Metric(T&&...) {}
 template<class... T> void Emit(T&&...) {}
};
class MemoryCensus;
inline void ReportSourceMemory(const SourceArtifact&,const char*) {}
inline void ReportGraphMemory(const LoweredProgram&,const char*) {}
inline void ReportAddressMemory(const ConstantAddressModel&,const char*) {}
inline void ReportTraceMemory(const ReferenceTrace&,const char*) {}
}
#endif
