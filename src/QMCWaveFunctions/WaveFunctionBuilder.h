#ifndef QMCPLUSPLUS_WAVEFUNCTION_BUILDER_H
#define QMCPLUSPLUS_WAVEFUNCTION_BUILDER_H
#include "Devices.h"
#include "Particle/ParticleSet.h"
#include "Utilities/RandomGenerator.h"

namespace qmcplusplus
{
class WaveFunction;

template<Devices DT>
class WaveFunctionBuilder
{
public:
static void build(bool useRef,
                        WaveFunction& WF,
                        ParticleSet& ions,
                        ParticleSet& els,
                        const RandomGenerator<QMCTraits::RealType>& RNG,
                        bool enableJ3);
};

extern template class WaveFunctionBuilder<Devices::CPU>;
}

#endif
