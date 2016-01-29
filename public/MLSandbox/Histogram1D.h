/* Copyright (C) 2013
 * Samuel Flis <samuel.flis@fysik.su.se>
 * Martin Wolf <martin.wolf@icecube.wisc.edu>
 * and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */
#ifndef MLSANDBOX_HISTOGRAM1D_H_INCLUDED
#define MLSANDBOX_HISTOGRAM1D_H_INCLUDED 1

#include <inttypes.h>
#include <math.h>

#include <iostream>

//==============================================================================
template<class T>
class Histogram1D
{
  public:
    Histogram1D();
    Histogram1D(uint64_t _bins, double _min, double _max);
    ~Histogram1D();

    uint64_t Fill(double value, T weight=1);
    double doubleFill(double value, T weight=1);
    bool SetBinContentByIndex(uint64_t index, T content);
    void Clear();
    T GetBinContentByIndex(uint64_t index) const;
    T GetBinContentByValue(double value) const;
    double IndexToValue(uint64_t index) const;
    uint64_t ValuteToIndex(T value) const;
    double GetBinWidth() const { return rangeOverBins; }
    uint64_t GetNBins() const { return bins; }
    double GetMin() const { return min; }
    double GetMax() const { return max; }
    std::vector<T> & GetHistogramArray() { return histogram; }
    std::vector<T> const & GetHistogramArray() const { return histogram; }
    T operator[](uint64_t index) const { return GetBinContentByIndex(index); }
    void Normalize(T to=1, bool density=true);
    void Scale(T by);
    void DivideByBinWidth();
    void Init(uint64_t _bins, double _min, double _max);
    T GetOverFlow() const {return overflow;}
    T GetUnderFlow() const {return underflow;}

    
  protected:
    uint64_t bins;
    double max;
    double min;
    double range;
    double binsOverRange; // bins/range
    double rangeOverBins;
    std::vector<T> histogram;
    double underflow;
    double overflow;
    bool init;
};
//==============================================================================

template<class T>
Histogram1D<T>::Histogram1D()
{
    init = true;
    min = 0.;
    max = 100.;
    range = max-min;
    bins = 100;
    binsOverRange = bins/range;
    rangeOverBins = range/bins;
    histogram.resize(bins);
}
//______________________________________________________________________________
template<class T>
void Histogram1D<T>::Init(uint64_t _bins, double _min, double _max){
    
    min = _min;
    max = _max;
    range = max-min;
    bins = _bins;
    binsOverRange = bins/range;
    rangeOverBins = range/bins;
    histogram.resize(bins);
}
//______________________________________________________________________________
template<class T>
Histogram1D<T>::Histogram1D(uint64_t _bins, double _min, double _max)
{
    Init(_bins, _min, _max);
}

//______________________________________________________________________________
template<class T>
Histogram1D<T>::~Histogram1D()
{
}

//______________________________________________________________________________
template<class T>
void Histogram1D<T>::DivideByBinWidth()
{
    double bin_width = this->GetBinWidth();
    for(uint64_t i = 0; i < bins; ++i) {
        histogram[i] /= bin_width;
    }
}

//______________________________________________________________________________
template<class T>
uint64_t Histogram1D<T>::Fill(double value, T weight)
{
    uint64_t index = (uint64_t)fabs(binsOverRange * (value - min));

    if(value < min) {
        underflow += weight;
        index = 0;
    }
    else if(index < bins){
        histogram[index] += weight;
        ++index;
    }
    else if(index >= bins) {
        overflow += weight;
        index = bins+1;
    }

    return index;
}

//______________________________________________________________________________
template<class T>
double Histogram1D<T>::IndexToValue(uint64_t index) const
{
//     if(index > bins)
//         index = bins;

    return  min + (int(index)-1)*rangeOverBins + rangeOverBins*0.5;
}
//______________________________________________________________________________
template<class T>
uint64_t Histogram1D<T>::ValuteToIndex(T value) const
{
//     if(value< max && value > min)
        
    return (uint64_t)fabs(binsOverRange * (value - min));
}
//______________________________________________________________________________
template<class T>
double Histogram1D<T>::doubleFill(double value, T weight)
{
    return (double)Fill(value, weight);
}

//______________________________________________________________________________
template<class T>
void Histogram1D<T>::Clear()
{
    for(uint64_t i = 0; i < bins; i++)
        histogram[i] = 0;
    underflow = 0;
    overflow = 0;
}

//____________________________________________________________________
template<class T>
T Histogram1D<T>::GetBinContentByIndex(uint64_t index) const
{
//     if(index <= bins) {
        return histogram[index-1];
//     }

    return 0;
}

//____________________________________________________________________
template<class T>
T Histogram1D<T>::GetBinContentByValue(double value) const
{
    uint64_t index = (uint64_t)fabs(binsOverRange * (value - min));

    if(index <= bins) {
        return histogram[index];
    }

    return 0;
}

//____________________________________________________________________
template<class T>
bool Histogram1D<T>::SetBinContentByIndex(uint64_t index, T content)
{
    if(index <= bins) {
        histogram[index-1] = content;
        return true;
    }

    return false;
}

//____________________________________________________________________
template<class T>
void Histogram1D<T>::Normalize(T to, bool density)
{
    T sum = 0;
    double binwidth = range/bins;

    for(uint64_t i = 0; i < bins; i++) {
        sum += histogram[i];
    }

    if(density)
        sum *= binwidth;

    for(uint64_t i = 0; i < bins; i++) {
        histogram[i] *= to/sum;
    }
}
//____________________________________________________________________
template<class T>
void Histogram1D<T>::Scale(T by)
{
    for(uint64_t i = 0; i < bins; i++) {
        histogram[i] *= by;
    }
}

#endif // MLSANDBOX_HISTOGRAM1D_H_INCLUDED
