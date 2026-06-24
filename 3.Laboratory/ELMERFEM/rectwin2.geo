//+
Point(1) = {0, 0, 0, 0.25};
//+
Point(2) = {2, 0, 0, 0.25};
//+
Line(1) = {1, 2};

Extrude{0,25,0} {
  Line{1};
}
//+
Physical Surface(6) = {5};
//+
Physical Curve(7) = {2};
//+
Physical Curve(8) = {4};
//+
Physical Curve(9) = {1};
//+
Physical Curve(10) = {3};
