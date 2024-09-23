/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"

#include <flow/flow_blackoil.hpp>

double ctime_stdw = 0.0;
double ctime_stdwperfrate = 0.0;
double ctime_stdwapply = 0.0;
double stdwapply_counter = 0.0;

double ctime_msw = 0.0;
double ctime_mswperfrate = 0.0;
double ctime_mswapply = 0.0;
double mswapply_counter = 0.0;
double ctime_mswdatatrans = 0.0;

int main(int argc, char** argv)
{
    return Opm::flowBlackoilTpfaMainStandalone(argc, argv);
}
