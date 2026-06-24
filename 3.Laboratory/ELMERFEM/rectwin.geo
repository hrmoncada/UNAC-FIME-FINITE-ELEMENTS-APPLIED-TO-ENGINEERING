ls = 0.25;
//+
Point(1) = {0, 0, 0, ls};
//+
Point(2) = {2, 0, 0, ls};
//+
Line(1) = {1, 2};

Extrude{0,25,0} {
  Line{1};
}
Physical Surface(6) = {5};
Physical Line(7) = {2};
Physical Line(8) = {4};
Physical Line(9) = {1};
Physical Line(10) = {3};

